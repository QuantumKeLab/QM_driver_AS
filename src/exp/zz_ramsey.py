from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from exp.RO_macros import multiRO_declare, multiRO_measurement, multiRO_pre_save
from qualang_tools.plot.fitting import Fit
# from common_fitting_func import *
import warnings
warnings.filterwarnings("ignore")
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
import xarray as xr
import time
import numpy as np

def exp_zz_ramsey(time_max,time_resolution,flux_range,ro_element,xy_element,con_xy_element,coupler_element,n_avg,config,qmm,virtual_detune=0,simulate:bool=False,initializer=None):
    """

    virtual_detune unit in MHz.\n
    time_max unit in us.\n
    time_resolution unit in us.\n
    """
    # v_detune_qua = virtual_detune *u.MHz
    # cc_resolution = (time_resolution/4.) *u.us
    # cc_max_qua = (time_max/4.) *u.us
    # cc_qua = np.arange( 4, cc_max_qua, cc_resolution)

    # evo_time = cc_qua*4
    # time_len = len(cc_qua)
    point_per_period = 20
    Ramsey_period = (1e3/virtual_detune)* u.ns
    tick_resolution = (Ramsey_period//(4*point_per_period))
    evo_time_tick_max = tick_resolution *point_per_period*6
    print(f"time resolution {tick_resolution*4} ,max time {evo_time_tick_max*4}")
    evo_time_tick = np.arange( 4, evo_time_tick_max, tick_resolution)
    evo_time = evo_time_tick*4
    time_len = len(evo_time)
    da = flux_range/5
    flux = np.arange(-flux_range, flux_range - da / 2, da)
    flux_len = len(flux) 
    with program() as ramsey:
        iqdata_stream = multiRO_declare( ro_element )
        n = declare(int)
        n_st = declare_stream()
        t = declare(int)  # QUA variable for the idle time, unit in clock cycle
        phi = declare(fixed)  # Phase to apply the virtual Z-rotation
        phi_idx = declare(bool,)
        X_idx = declare(bool,)
        dc = declare(fixed) 
        with for_(n, 0, n < n_avg, n + 1):
            with for_each_( X_idx, [True, False]):
                with for_each_( phi_idx, [True, False]):
                    with for_(*from_array(dc, flux)):
                        with for_( *from_array(t, evo_time_tick) ):
                            # Init
                            if initializer is None:
                                wait(100*u.us)
                            else:
                                try:
                                    initializer[0](*initializer[1])
                                except:
                                    print("Initializer didn't work!")
                                    wait(100*u.us)

                            # Operation
                            True_value = Cast.mul_fixed_by_int(virtual_detune * 1e-3, 4 * t)
                            False_value = Cast.mul_fixed_by_int(-virtual_detune * 1e-3, 4 * t)
                            assign(phi, Util.cond(phi_idx, True_value, False_value))
                            # phi = Cast.mul_fixed_by_int( virtual_detune/1e3, 4 *cc)
                            # True_value =  v_detune_qua*4*cc
                            # False_value = v_detune_qua*4*cc

                            with if_(X_idx):
                                for con_xy in con_xy_element:
                                    play("x180", con_xy)   # conditional x180 gate
                                align()

                            for xy in xy_element:
                                play("x90", xy)  # 1st x90 gate
                                play("const"*amp(dc*10.), coupler_element, t)         #const 預設0.1
                                wait(t, xy)
                                frame_rotation_2pi(phi, xy)  # Virtual Z-rotation
                                play("x90", xy)  # 2st x90 gate

                            # Align after playing the qubit pulses.
                            align()
                            # Readout
                            multiRO_measurement(iqdata_stream, ro_element, weights="rotated_")         
                        

            # Save the averaging iteration to get the progress bar
            save(n, n_st)

        with stream_processing():
            n_st.save("iteration")
            multiRO_pre_save(iqdata_stream, ro_element, (2, 2, flux_len, time_len) )

    ###########################
    # Run or Simulate Program #
    ###########################


    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=20_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, ramsey, simulation_config)
        job.get_simulated_samples().con1.plot()
        job.get_simulated_samples().con2.plot()
        plt.show()

    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(ramsey)
        # Get results from QUA program
        ro_ch_name = []
        for r_name in ro_element:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")
        data_list = ro_ch_name + ["iteration"]   
        results = fetching_tool(job, data_list=data_list, mode="live")
        # Live plotting

        fig, ax = plt.subplots(2, len(ro_element))
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        fig.suptitle("Frequency calibration")

        # Live plotting
        while results.is_processing():
            # Fetch results
            fetch_data = results.fetch_all()
            # Progress bar
            iteration = fetch_data[-1]
            progress_counter(iteration, n_avg, start_time=results.start_time)
            # Plot
            plt.tight_layout()
            time.sleep(1)

        # Measurement finished 
        fetch_data = results.fetch_all()
        qm.close()
        output_data = {}

        for r_idx, r_name in enumerate(ro_element):
            output_data[r_name] = ( ["mixer","X","frequency","flux","time"],
                                np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) )
        dataset = xr.Dataset(
            output_data,
            coords={ "mixer":np.array(["I","Q"]), "X": np.array([True, False]), "frequency": np.array([virtual_detune,-virtual_detune]), "flux":  flux, "time": evo_time}
        )

        return dataset
       
def T2_fitting(signal):
    try:
        fit = Fit()
        decay_fit = fit.ramsey(4 * idle_times, signal, plot=False)
        qubit_T2 = np.round(np.abs(decay_fit["T2"][0]) / 4) * 4
    except Exception as e:     
        print(f"An error occurred: {e}")  
        qubit_T2 = 0
    return qubit_T2

def multi_T2_exp(m, Qi, n_avg,idle_times,operation_flux_point,q_id,qmm):
    T2_I, T2_Q = [], []
    for i in range(m):
        I, Q = exp_zz_ramsey(Qi,n_avg,idle_times,operation_flux_point,q_id,qmm)
        T2_I.append(T2_fitting(I))
        T2_Q.append(T2_fitting(Q))
        print(f'iteration: {i+1}')
    return T2_I, T2_Q

def plot_ramsey_oscillation( x, y, ax=None ):
    """
    y in shape (2,N)
    2 is postive and negative
    N is evo_time_point
    """
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(x, y, "-",label="T2")
    ax.set_xlabel("Free Evolution Times [ns]")
    ax.legend()
    if ax == None:
        return fig
    
def choose_idata(dataset):
    for ro_name, data in dataset.data_vars.items():
        idata = data[0]
    return idata
    
def plot_phase(x, dataset1, dataset2, ax=None):
    """
    x in shape (N, )
    dataset in shape (2,N)
    dataset1 w/o conditional x180 gate
    dataset2 w/ conditional x180 gate
    2 is I and Q
    N is evo_time_point
    """
    idata1 = choose_idata(dataset1)
    idata2 = choose_idata(dataset2)
    if ax == None:
        fig, ax = plt.subplots()
    try:
        fit = Fit()
        decay_fit1 = fit.ramsey(4 * x, idata1, plot=False)
        freq1 = decay_fit1["f"][0]
        print("w/o X freq:")
        print(freq1)
    except:
        print("an error occured in data1")
        freq1 = 0
    try:
        fit = Fit()
        decay_fit2 = fit.ramsey(4 * x, idata2, plot=False)
        freq2 = decay_fit2["f"][0]
        print("w X freq:")
        print(freq2)
    except:
        print("an error occured in data2")
        freq2 = 0
    phase1 = 2*np.pi*freq1*x
    phase2 = 2*np.pi*freq2*x
    
    # ax.plot(x,idata1,"-",color="b", label="w/ x180 gate")
    # ax.plot(x,idata2,"--",color="r", label="w/o x180 gate")
    ax.plot(x,phase1,"-",color="b", label="w/o x180 gate")
    ax.plot(x,phase2,"--",color="r", label="w/ x180 gate")
    ax.set_xlabel("Free Evolution Times [ns]")
    if ax == None:
        return fig

def plot_difference(x, dataset1, dataset2, ax=None):
    """
    x in shape (N, )
    dataset in shape (2,N)
    dataset1 w/o conditional x180 gate
    dataset2 w/ conditional x180 gate
    2 is I and Q
    N is evo_time_point
    """
    idata1 = choose_idata(dataset1)
    idata2 = choose_idata(dataset2)
    if ax == None:
        fig, ax = plt.subplots()
    # try:
    #     fit = Fit()
    #     decay_fit1 = fit.ramsey(4 * idle_times, idata1, plot=False)
    #     freq1 = decay_fit1["f"][0]
    # except:
    #     print("an error occured in data1")
    #     freq1 = 0
    # try:
    #     fit = Fit()
    #     decay_fit2 = fit.ramsey(4 * idle_times, idata2, plot=False)
    #     freq2 = decay_fit2["f"][0]
    # except:
    #     print("an error occured in data2")
    #     freq2 = 0
    # phase1 = 2*np.pi*freq1*x
    # phase2 = 2*np.pi*freq2*x
    ax.plot(x,idata1,"-",color="b", label="w/o x180 gate")
    ax.plot(x,idata2,"--",color="r", label="w/ x180 gate")
    # ax.plot(x,phase1,"-",color="b", label="w/o x180 gate")
    # ax.plot(x,phase2,"--",color="r", label="w/ x180 gate")
    ax.set_xlabel("Free Evolution Times [ns]")
    if ax == None:
        return fig



def plot_ana_result( evo_time, flux, detuning, data, ax=None ):
    """
    data in shape (2,N)
    2 is postive and negative
    N is evo_time_point
    """
    if ax == None:
        fig, ax = plt.subplots()
    fit = Fit()
    frequency_X=np.zeros(len(flux))
    frequency_I=np.zeros(len(flux))
    for i in range(len(flux)):
        print(i)
        ana_dict_pos = fit.ramsey(evo_time, data[1][0][i], plot=False)
        ana_dict_neg = fit.ramsey(evo_time, data[1][1][i], plot=False)
        freq_pos = ana_dict_pos['f'][0]*1e3
        freq_neg = ana_dict_neg['f'][0]*1e3
        frequency_I[i] = (freq_pos-freq_neg)/2
        ana_dict_pos = fit.ramsey(evo_time, data[0][0][i], plot=False)
        ana_dict_neg = fit.ramsey(evo_time, data[0][1][i], plot=False)
        freq_pos = ana_dict_pos['f'][0]*1e3
        freq_neg = ana_dict_neg['f'][0]*1e3
        frequency_X[i] = (freq_pos-freq_neg)/2

    ax.set_xlabel("flux [mV]")


    ax.plot(flux, frequency_I, label=f"w/o X")#: {freq_pos:.3f} MHz")
    ax.plot(flux, frequency_X, label=f"w/ X")#: {freq_neg:.3f} MHz")
    ax.plot(flux, frequency_X-frequency_I, label=f"diff")
    ax.text(0.07, 0.9, f"min diff at : {flux[np.argmin(frequency_X-frequency_I)]:.3f}", fontsize=10, transform=ax.transAxes)

    ax.legend()
    plt.tight_layout()

# def T2_hist(data, T2_max, signal_name):
#     try:
#         new_data = [x / 1000 for x in data]  
#         bin_width = 0.5  
#         start_value = -0.25  
#         end_value = T2_max + 0.25  
#         custom_bins = [start_value + i * bin_width for i in range(int((end_value - start_value) / bin_width) + 1)]
#         hist_values, bin_edges = np.histogram(new_data, bins=custom_bins, density=True)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         params, covariance = curve_fit(gaussian, bin_centers, hist_values)
#         mu, sigma = params
#         plt.cla()
#         plt.hist(new_data, bins=custom_bins, density=True, alpha=0.7, color='blue', label='Histogram') 
#         xmin, xmax = plt.xlim()
#         x = np.linspace(xmin, xmax, 100)
#         p = gaussian(x, mu, sigma)
#         plt.plot(x, p, 'k', linewidth=2, label=f'Fit result: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
#         plt.legend()
#         plt.title('T2_'+signal_name+' Gaussian Distribution Fit')
#         plt.show()
#         print(f'Mean: {mu:.2f}')
#         print(f'Standard Deviation: {sigma:.2f}')
#     except Exception as e:
#         print(f"An error occurred: {e}")


if __name__ == '__main__':
    from configuration import *

    n_avg = 750
    idle_times = np.arange(4, 200, 1)  
    detuning = 1e6  
    operation_flux_point = [0, -3.000e-01, -0.2525, -0.3433, -3.400e-01] 
    q_id = [0,1,2,3]
    Qi = 3

    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)
    I,Q = T2_exp(Qi,n_avg,idle_times,operation_flux_point,q_id,qmm)
    T2_plot(I, Q, Qi, True)
    # m = 3
    # T2_I, T2_Q = multi_T2_exp(m, Qi, n_avg,idle_times,operation_flux_point,q_id,qmm)
    # T2_hist(T2_I,15,'I')
    # T2_hist(T2_Q,15,'Q')
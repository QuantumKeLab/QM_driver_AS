"""
        TIME RABI
The sequence consists in playing the qubit pulse and measuring the state of the resonator
for different qubit pulse durations.
The results are then post-processed to find the qubit pulse duration for the chosen amplitude.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse amplitude (rabi_chevron_amplitude or power_rabi).
    - Set the qubit frequency and desired pi pulse amplitude (pi_amp_q) in the configuration.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse duration (pi_len_q) in the configuration.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import sys
import pathlib
#QM_script_root = str(pathlib.Path(__file__).parent.parent.resolve())
#sys.path.append(QM_script_root)
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from exp.RO_macros import state_tomo_singleRO_declare, tomo_pre_save_singleShot, state_tomo_measurement, tomo_NQ_proj
import warnings
import xarray as xr
import numpy as np

warnings.filterwarnings("ignore")
from exp.QMMeasurement import QMMeasurement

class StateTomography(QMMeasurement):
    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q4_ro"]
        self.xy_elements = ["q4_xy"]
        
        self.initializer = None
        #self.process = self.prepare_state
    
    def _get_qua_program(self):
        with program() as tomo:
            iqdata_stream = state_tomo_singleRO_declare( self.ro_elements )
            n = declare(int)  # QUA variable for the qubit pulse duration
            n_st = declare_stream()

            with for_(n, 0, n < self.shot_num, n + 1):

                tomo_NQ_proj( iqdata_stream, self.process, self.xy_elements, self.ro_elements, weights="rotated_", thermalization_time=100)

                # Wait for the qubit to decay to the ground state
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                n_st.save("n")
                tomo_pre_save_singleShot( iqdata_stream, self.xy_elements, self.ro_elements )
        return tomo
    
    def _get_fetch_data_list( self ):
        ro_ch_name = []
        for r_name in self.ro_elements:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")

        data_list = ro_ch_name + ["n"] 
        return data_list
    
    def _data_formation( self ):
        output_data = {}
        coords={"mixer":np.array(["I","Q"]), "shot":np.arange(self.shot_num)}
        dims_order = ["mixer","shot"]
        for i in range(len(self.xy_elements)):
            dims_order.append(f"basis{i+1}")
            coords[f"basis{i+1}"] = ["z","x","y"]

        for r_idx, r_name in enumerate(self.ro_elements):
            output_data[r_name] = ( dims_order,
                np.squeeze(np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) ))
            dataset = xr.Dataset(
                output_data,
                coords
            )
        # Live plotting
        return dataset
    
    def process(self):
        
        #play("x180", "q3_xy")
        play("x90", "q4_xy" )

        pass

    

def state_tomography( q_name, ro_element, prepare_state, n_avg, config, qmm:QuantumMachinesManager, simulate:bool=True):
    
    with program() as tomo:
        iqdata_stream = state_tomo_singleRO_declare( ro_element )
        n = declare(int)  # QUA variable for the qubit pulse duration
        n_st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):

            state_tomo_measurement( iqdata_stream, prepare_state, q_name, ro_element, weights="rotated_", thermalization_time= 200)

            # Wait for the qubit to decay to the ground state
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

        with stream_processing():
            n_st.save("n")
            tomo_pre_save_singleShot( iqdata_stream, q_name, ro_element )

    ###########################
    # Run or Simulate Program #
    ###########################
    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, tomo, simulation_config)
        job.get_simulated_samples().con1.plot()
        job.get_simulated_samples().con2.plot()
        plt.show()

    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(tomo)

        data_list = []
        projection_type = ["x", "y", "z"]

        for r in ro_element:
            data_list.append(f"{r}_I")
            data_list.append(f"{r}_Q")

        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, data_list, mode="wait_for_all")
        fetch_data = results.fetch_all()
        qm.close()

        output_data = {}
        for r_idx, r_name in enumerate(ro_element):
            output_data[r_name] = ( ["mixer","shot","basis"],
                np.squeeze(np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) ))
            dataset = xr.Dataset(
                output_data,
                coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(n_avg),"basis":["z","x","y"] }
            )
        # Live plotting

        return dataset


def state_tomography_NQ( q_name, ro_element, prepare_state, n_avg, config, qmm:QuantumMachinesManager, simulate:bool=True):
    with program() as tomo:
        iqdata_stream = state_tomo_singleRO_declare( ro_element )
        n = declare(int)  # QUA variable for the qubit pulse duration
        n_st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):

            tomo_NQ_proj( iqdata_stream, prepare_state, q_name, ro_element, weights="rotated_", thermalization_time=200)

            # Wait for the qubit to decay to the ground state
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

        with stream_processing():
            n_st.save("n")
            tomo_pre_save_singleShot( iqdata_stream, q_name, ro_element )

    ###########################
    # Run or Simulate Program #
    ###########################
    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, tomo, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show()

    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(tomo)

        data_list = []
        projection_type = ["x", "y", "z"]

        for r in ro_element:
            data_list.append(f"{r}_I")
            data_list.append(f"{r}_Q")

        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, data_list, mode="wait_for_all")
        fetch_data = results.fetch_all()
        qm.close()

        output_data = {}
        coords={"mixer":np.array(["I","Q"]), "shot":np.arange(n_avg)}
        dims_order = ["mixer","shot"]
        for i in range(len(ro_element)):
            dims_order.append(f"basis{i+1}")
            coords["basis%d"%(i+1)] = ["z","x","y"]

        for r_idx, r_name in enumerate(ro_element):
            output_data[r_name] = ( dims_order,
                np.squeeze(np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) ))
            dataset = xr.Dataset(
                output_data,
                coords
            )
        # Live plotting
        return dataset

def calculate_block_vector( data, threshold ):
    vect_dis = []
    dirction = ["z", "x", "y"]
    total_count = data.shape[-1]
    for idx, dir in enumerate(dirction):
        count_0 = np.count_nonzero(data[idx] > threshold)
        pop_0 = count_0/total_count
        print(f"{dir} g count {total_count}, {pop_0}")

        pop_1 = 1-pop_0
        vect_dis.append(pop_0-pop_1)
    vect_dis = [vect_dis[1],vect_dis[2],vect_dis[0]]
    return vect_dis

def calculate_2Q_block_vector(data1, threshold1, data2, threshold2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    vect_dis = []
    dirction = ["z", "x", "y"]
    total_count = data1.shape[-1]
    for idx1, dir1 in enumerate(dirction):
        for idx2, dir2 in enumerate(dirction):
            count_0 = np.count_nonzero(np.all([data1[idx1,idx2] < threshold1,data2[idx1,idx2] < threshold2],axis=0))
            pop_0 = count_0/total_count
            print(f"{dir1}{dir2} g count {total_count}, {pop_0}")

            pop_1 = 1-pop_0
            vect_dis.append(pop_0-pop_1)
    vect_dis = [vect_dis[1],vect_dis[2],vect_dis[0]]
    return vect_dis

            

def calculate_density_matrix(data1, threshold1, data2, threshold2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    density_matrix = np.zeros((4,4))
    dirction = ["z", "x", "y"]
    basis_coeffient_matrix = np.zeros((4,4))
    basis = ["i","z","x","y"]
    total_count = data1.shape[-1]
    for idx1, dir1 in enumerate(dirction):
        for idx2, dir2 in enumerate(dirction):
            count_00 = np.count_nonzero(np.all([data1[idx1,idx2] <= threshold1,data2[idx1,idx2] <= threshold2],axis=0))
            count_10 = np.count_nonzero(np.all([data1[idx1,idx2] > threshold1,data2[idx1,idx2] <= threshold2],axis=0))
            count_01 = np.count_nonzero(np.all([data1[idx1,idx2] <= threshold1,data2[idx1,idx2] > threshold2],axis=0))
            count_11 = np.count_nonzero(np.all([data1[idx1,idx2] > threshold1,data2[idx1,idx2] > threshold2],axis=0))
            pop_00 = count_00/total_count
            pop_10 = count_10/total_count
            pop_01 = count_01/total_count
            pop_11 = count_11/total_count
            if dir1 != dir2:
                basis_coeffient_matrix[idx1+1,idx2+1] = pop_00-pop_10-pop_01+pop_11
            else:
                basis_coeffient_matrix[idx1+1,idx2+1] = pop_00-pop_10-pop_01+pop_11
                basis_coeffient_matrix[0,idx2+1] = pop_00+pop_10-pop_01-pop_11
                basis_coeffient_matrix[idx1+1,0] = pop_00-pop_10+pop_01-pop_11
    basis_coeffient_matrix[0,0] = 1

    pauli_x = np.array([[0,1],[1,0]])
    pauli_y = np.array([[0,-1j],[1j,0]])
    pauli_z = np.array([[1,0],[0,-1]])
    idendity = np.eye(2)
    single_q_basis = [idendity, pauli_z, pauli_x, pauli_y]
    two_q_basis = []
    for basis_1 in single_q_basis:
        for basis_2 in single_q_basis:
            two_q_basis.append(np.kron(basis_1, basis_2))
    two_q_basis = np.array(two_q_basis).reshape((4,4,4,4))
    for idx1, dir1 in enumerate(basis):
        for idx2, dir2 in enumerate(basis):
            density_matrix = density_matrix+basis_coeffient_matrix[idx1,idx2]*two_q_basis[idx1,idx2]
    density_matrix = 1/4*density_matrix

    return density_matrix

def plot_density_matrix(density_matrix):
    #from qutip.visualization import matrix_histogram_complex
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(projection='3d')
    density_matrix_1D = density_matrix.reshape((16,))
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap('hsv')
    rgba = [cmap((np.angle(k)+np.pi)/(2*np.pi)) for k in density_matrix_1D]
    x = np.array([1,3,5,7]*4)
    y = np.array([1]*4+[3]*4+[5]*4+[7]*4)
    z = 0
    phase_map = np.abs(np.angle(density_matrix))/(2*np.pi)
    ax.bar3d(x,y,z,dx=1,dy=1,dz=np.abs(density_matrix_1D),color=rgba)
    plt.xticks(ticks=[1,3,5,7],labels=["|00>","|01>","|10>","|11>"])
    plt.yticks(ticks=[1,3,5,7],labels=["|00>","|01>","|10>","|11>"])
    plt.title("Density matrix of Bell state")
    sm = cm.ScalarMappable(norm=norm,cmap=cmap)
    sm.set_array([])
    print(np.angle(density_matrix))
    print([np.angle(k) for k in density_matrix_1D])
    plt.colorbar(sm,ax=ax).set_label("Phase (pi)")

def plot_block_vector( vect_dis, fig=None ):
    from qutip import Bloch
    if fig == None:
        fig = Bloch()
    fig.add_vectors(vect_dis)
    return fig

if __name__ == '__main__':
    ro_element = ["rr1"]
    n_avg = 10000
    q_name = ["q1_xy"]
    threshold = ge_threshold_q1

    #####################################
    #  Open Communication with the QOP  #
    #####################################e4
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

    def prepare_state():

        play("x90", "q1_xy" )
        # play("x180", "q2_xy" )

        pass

    # data = state_tomography(q_name, ro_element, prepare_state, n_avg, config, qmm, False)
    # print(data)

    data = state_tomography_NQ(q_name, ro_element, prepare_state, n_avg, config, qmm, False)
    print(type(data),data.keys())

    # np.savez("11.npz",**data)
    # data = np.load("00.npz")
    # for label in data.keys():
    #     print(data[label].shape)
    # Plot Bloch Sphere

    # data = data[ro_element[0]]
    # print(data.shape)

    # total_count = n_avg
    # print(f"total count {total_count}")
    # dirction = ["x","y","z"]
    data_i = data[ro_element[0]][0].transpose()
    data_q = data[ro_element[0]][1].transpose()
    # plt.plot(data_i[0],data_q[0],'o')

    vect_dis = calculate_block_vector(data_i, threshold)
    block_plot = plot_block_vector(vect_dis)
    block_plot.show()
    plt.show()

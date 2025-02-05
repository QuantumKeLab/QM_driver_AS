from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from exp.RO_macros import multiRO_declare, multiRO_measurement, multiRO_pre_save
from qualang_tools.units import unit
import numpy as np
import time
import xarray as xr
import exp.config_par as gc
import warnings
from exp.QMMeasurement import QMMeasurement
u = unit(coerce_to_integer=True)
warnings.filterwarnings("ignore")


class CZChavron(QMMeasurement):
    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q2_ro","q3_ro"]
        self.flux_Qi = 3
        self.excited_Qi = [2,3]
        self.flux_Ci = 7
        
        self.mode = "coupler-qubit" #coupler-qubit, qubit-time, time-coupler
        self.preprocess = "ave"
        self.initializer = None
        
        self.time_max = 1 #us
        self.time_resolution = 0.004 # us
        self.z_amps_range = ( -1, 1 )
        self.z_amps_resolution = 0.1
        self.couplerz_amps_range = (-0.5, 0.5)
        self.couplerz_amps_resolution = 0.05

        self.time = 60 # ns
        self.z_amps = 0.3
        self.couplerz_amps = 0.2

    def _get_qua_program( self ):
        
        self.qua_z_amp_ratio_array = self._lin_z_amp_array()
        self.qua_couplerz_amp_ratio_array = self._lin_couplerz_amp_array()

        self.qua_evo_time = self._lin_evo_time_array()
        self._attribute_config()
        match self.mode:
            case "qubit-time":
                with program() as cz_chavron:

                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)  
                    n_st = declare_stream()
                    cc = declare(int)  
                    z_amps = declare(fixed)  

                    with for_(n, 0, n < self.shot_num, n + 1):
                        with for_(*from_array(z_amps, self.qua_z_amp_ratio_array)):
                            with for_(*from_array(cc,self.qua_evo_time )):
                                # initializaion
                                if self.initializer is None:
                                    wait(1*u.us,self.ro_elements)
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        wait(1*u.us,self.ro_elements)
                                
                                # operation
                                if self.excited_Qi != []: 
                                    for excited_Qi in self.excited_Qi:
                                        play("x180", f"q{excited_Qi}_xy")
                                align()
                                wait(20 *u.ns)
                                play("const" * amp(z_amps), f"q{self.flux_Qi}_z", duration=cc)
                                play("const" * amp(self.couplerz_amps),f"q{self.flux_Ci}_z",duration=cc) 
                                wait(20 *u.ns)
                                #play( "const"*amp( -0.2 ), "q7_z", duration=100)
                                wait(cc)
                                multiRO_measurement(iqdata_stream, self.ro_elements, weights="rotated_") 
                        save(n, n_st)

                    with stream_processing():
                        n_st.save("iteration")
                        match self.preprocess:
                            case "shot":
                                multiRO_pre_save(iqdata_stream, self.ro_elements, (self.shot_num,len(self.qua_z_amp_ratio_array),len(self.qua_evo_time)) ,stream_preprocess="shot")
                            case _:
                                multiRO_pre_save(iqdata_stream, self.ro_elements, (len(self.qua_z_amp_ratio_array),len(self.qua_evo_time)))

                return cz_chavron
            
            case "coupler-qubit":
                qua_couplerz_time = self.time//4
                qua_z_time = self.time//4 #+ 10
                with program() as cz_chavron:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()
                    couplerz_amps = declare(fixed)  
                    z_amps = declare(fixed)
                    with for_(n, 0, n < self.shot_num, n + 1):
                        with for_(*from_array(couplerz_amps, self.qua_couplerz_amp_ratio_array)):
                            with for_(*from_array(z_amps,self.qua_z_amp_ratio_array )):
                                # initializaion
                                if self.initializer is None:
                                    wait(1*u.us,self.ro_elements)
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        wait(1*u.us,self.ro_elements)
                                
                                # operation
                                if self.excited_Qi != []: 
                                    for excited_Qi in self.excited_Qi:
                                        play("x180", f"q{excited_Qi}_xy")
                                align()
                                wait(20 *u.ns)
                                play("const" * amp(z_amps), f"q{self.flux_Qi}_z", duration=qua_z_time)
                                #wait(20 *u.ns,f"q{self.flux_Ci}_z")
                                play("const" * amp(couplerz_amps), f"q{self.flux_Ci}_z", duration=qua_couplerz_time)
                                align()               
                                wait(20 * u.ns)
                                #play( "const"*amp( -0.2 ), "q7_z", duration=100)
                                multiRO_measurement(iqdata_stream, self.ro_elements, weights="rotated_") 
                        save(n, n_st)

                    with stream_processing():
                        n_st.save("iteration")
                        match self.preprocess:
                            case "shot":
                                multiRO_pre_save(iqdata_stream, self.ro_elements, (self.shot_num,len(self.qua_couplerz_amp_ratio_array),len(self.qua_z_amp_ratio_array)) ,stream_preprocess="shot")
                            case _:
                                multiRO_pre_save(iqdata_stream, self.ro_elements, (len(self.qua_couplerz_amp_ratio_array),len(self.qua_z_amp_ratio_array)))
                return cz_chavron
            case "time-coupler":
                with program() as cz_chavron:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()
                    cc = declare(int)  
                    couplerz_amps = declare(fixed)
                    with for_(n, 0, n < self.shot_num, n + 1):
                        with for_(*from_array(cc,self.qua_evo_time )):
                            with for_(*from_array(couplerz_amps, self.qua_couplerz_amp_ratio_array)):
                                # initializaion
                                if self.initializer is None:
                                    wait(1*u.us,self.ro_elements)
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        wait(1*u.us,self.ro_elements)
                                
                                # operation
                                if self.excited_Qi != []: 
                                    for excited_Qi in self.excited_Qi:
                                        play("x180", f"q{excited_Qi}_xy")
                                align()
                                wait(20 *u.ns)
                                play("const" * amp(self.z_amps), f"q{self.flux_Qi}_z", duration=cc)
                                play("const" * amp(couplerz_amps),f"q{self.flux_Ci}_z",duration=cc) 
                                wait(20 *u.ns)
                                #play( "const"*amp( -0.2 ), "q7_z", duration=100)
                                wait(cc)
                                multiRO_measurement(iqdata_stream, self.ro_elements, weights="rotated_") 
                        save(n, n_st)

                    with stream_processing():
                        n_st.save("iteration")
                        match self.preprocess:
                            case "shot":
                                multiRO_pre_save(iqdata_stream, self.ro_elements, (self.shot_num,len(self.qua_evo_time),len(self.qua_couplerz_amp_ratio_array)) ,stream_preprocess="shot")
                            case _:
                                multiRO_pre_save(iqdata_stream, self.ro_elements, (len(self.qua_evo_time),len(self.qua_couplerz_amp_ratio_array)))
                return cz_chavron
    def _get_fetch_data_list( self ):
        ro_ch_name = []
        for r_name in self.ro_elements:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")

        data_list = ro_ch_name + ["iteration"]   
        return data_list
    
    def _data_formation( self ):
        output_data = {}
        match self.mode:
            case "qubit-time":
                match self.preprocess:
                    case "shot":
                        for r_idx, r_name in enumerate(self.ro_elements):
                            output_data[r_name] = ( ["mixer","shot","amps","time"],
                                        np.squeeze(np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) ))
                        dataset = xr.Dataset(
                            output_data,
                            coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(self.shot_num),"amps":self.qua_z_amp_ratio_array, "time": self.evo_time }
                        )
                    case _:
                        for r_idx, r_name in enumerate(self.ro_elements):
                            output_data[r_name] = ( ["mixer","amps","time"],
                                        np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
                        dataset = xr.Dataset(
                            output_data,
                            coords={ "mixer":np.array(["I","Q"]), "amps":self.qua_z_amp_ratio_array, "time": self.evo_time }
                        )
            case "coupler-qubit":
                match self.preprocess:
                    case "shot":
                        for r_idx, r_name in enumerate(self.ro_elements):
                            output_data[r_name] = ( ["mixer","shot","c_amps","amps"],
                                        np.squeeze(np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) ))
                        dataset = xr.Dataset(
                            output_data,
                            coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(self.shot_num),"c_amps":self.qua_couplerz_amp_ratio_array, "amps": self.qua_z_amp_ratio_array}
                        )
                    case _:
                        for r_idx, r_name in enumerate(self.ro_elements):
                            output_data[r_name] = ( ["mixer","c_amps","amps"],
                                        np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
                        dataset = xr.Dataset(
                            output_data,
                            coords={ "mixer":np.array(["I","Q"]), "c_amps": self.qua_couplerz_amp_ratio_array, "amps": self.qua_z_amp_ratio_array }
                        )
            case "time-coupler":
                match self.preprocess:
                    case "shot":
                        for r_idx, r_name in enumerate(self.ro_elements):
                            output_data[r_name] = ( ["mixer","shot","time","c_amps"],
                                        np.squeeze(np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) ))
                        dataset = xr.Dataset(
                            output_data,
                            coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(self.shot_num), "time": self.evo_time, "c_amps":self.qua_couplerz_amp_ratio_array}
                        )
                    case _:
                        for r_idx, r_name in enumerate(self.ro_elements):
                            output_data[r_name] = ( ["mixer","time","c_amps"],
                                        np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
                        dataset = xr.Dataset(
                            output_data,
                            coords={ "mixer":np.array(["I","Q"]), "time": self.evo_time, "c_amps":self.qua_couplerz_amp_ratio_array}
                        )
        
        self._attribute_config()
        dataset.attrs["ro_LO"] = self.ref_ro_LO
        dataset.attrs["ro_IF"] = self.ref_ro_IF
        dataset.attrs["xy_LO"] = self.ref_xy_LO
        dataset.attrs["xy_IF"] = self.ref_xy_IF
        dataset.attrs["z_offset"] = self.z_offset

        dataset.attrs["z_amp_const"] = self.z_amp
        return dataset
    
    def _attribute_config( self ):
        self.ref_ro_IF = []
        self.ref_ro_LO = []
        for r in self.ro_elements:
            self.ref_ro_IF.append(gc.get_IF(r, self.config))
            self.ref_ro_LO.append(gc.get_LO(r, self.config))

        self.ref_xy_IF = []
        self.ref_xy_LO = []
        for xy in self.excited_Qi:
            self.ref_xy_IF.append(gc.get_IF(f"q{xy}_xy", self.config))
            self.ref_xy_LO.append(gc.get_LO(f"q{xy}_xy", self.config))

        self.z_offset = []
        self.z_offset.append( gc.get_offset(f"q{self.flux_Qi}_z", self.config ))
        self.z_offset.append( gc.get_offset(f"q{self.flux_Ci}_z", self.config ))
        self.z_amp = []
        self.z_amp.append(gc.get_const_wf(f"q{self.flux_Qi}_z", self.config ))
        self.z_amp.append(gc.get_const_wf(f"q{self.flux_Ci}_z", self.config ))
        
    
    def _lin_z_amp_array( self ):
        amp_ratio = np.arange( self.z_amps_range[0],self.z_amps_range[1], self.z_amps_resolution)
        return amp_ratio
    def _lin_couplerz_amp_array(self):
        amp_ratio = np.arange( self.couplerz_amps_range[0],self.couplerz_amps_range[1], self.couplerz_amps_resolution)
        return amp_ratio
    
    def _lin_evo_time_array(self):
        cc_max_qua = (self.time_max/4) * u.us
        cc_resolution_qua = (self.time_resolution/4) * u.us
        cc_delay_qua = np.arange( 4, cc_max_qua, cc_resolution_qua)
        self.evo_time = cc_delay_qua*4
        return cc_delay_qua

def CZ(time_max,time_resolution,z_amps_range,z_amps_resolution,ro_element,flux_Qi,excited_Qi,flux_Ci,coupler_z,preprocess,qmm,config,n_avg=1,initializer=None,simulate=True):
    cc_resolution = (time_resolution/4.)*u.us
    cc_max_qua = (time_max/4.)*u.us
    cc_qua = np.arange( 4, cc_max_qua, cc_resolution)
    print(cc_qua)
    evo_time = cc_qua*4
    
    z_amps_array = np.arange(z_amps_range[0],z_amps_range[1],z_amps_resolution)
    amp_len = len(z_amps_array)
    time_len = len(evo_time)

    with program() as cz:
        iqdata_stream = multiRO_declare( ro_element )
        n = declare(int)
        n_st = declare_stream()
        cc = declare(int)  
        z_amps = declare(fixed)
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(z_amps, z_amps_array)):
                with for_(*from_array(cc,cc_qua )):
                    # initializaion
                    if initializer is None:
                        wait(1*u.us,ro_element)
                    else:
                        try:
                            initializer[0](*initializer[1])
                        except:
                            wait(1*u.us,ro_element)
                    
                    # operation
                    if excited_Qi != []: 
                        for excited_Qi in excited_Qi:
                            play("x180", f"q{excited_Qi}_xy")
                    align()
                    wait(20 *u.ns)
                    play("const" * amp(z_amps), f"q{flux_Qi}_z", duration=cc)
                    play("const" * amp(coupler_z),f"q{flux_Ci}_z",duration=cc) 
                    wait(20 *u.ns)
                    play( "const"*amp( -0.4 ), "q7_z", duration=100)
                    wait(cc)
                    multiRO_measurement(iqdata_stream, ro_element, weights="rotated_") 
            save(n, n_st)

        with stream_processing():
            n_st.save("iteration")
            match preprocess:
                case "shot":
                    multiRO_pre_save(iqdata_stream, ro_element, (n_avg,amp_len,time_len) ,stream_preprocess="shot")
                case _:
                    multiRO_pre_save(iqdata_stream, ro_element, (amp_len,time_len))
    if simulate:
        simulation_config = SimulationConfig(duration=20000)  # In clock cycles = 4ns
        job = qmm.simulate(config, cz, simulation_config)
        job.get_simulated_samples().con1.plot()
        #job.get_simulated_samples().con2.plot()
        plt.show()
    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(cz)
        # Get results from QUA program
        ro_ch_name = []
        for r_name in ro_element:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")
        data_list = ro_ch_name + ["iteration"]   
        results = fetching_tool(job, data_list=data_list, mode="live")

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
        match preprocess:
            case "shot":
                for r_idx, r_name in enumerate(ro_element):
                    output_data[r_name] = ( ["mixer","shot","amps","time"],
                                np.squeeze(np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) ))
                dataset = xr.Dataset(
                    output_data,
                    coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(n_avg),"amps":z_amps_array, "time": evo_time }
                )
            case _:
                for r_idx, r_name in enumerate(ro_element):
                    output_data[r_name] = ( ["mixer","amps","time"],
                                np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) )
                dataset = xr.Dataset(
                    output_data,
                    coords={ "mixer":np.array(["I","Q"]), "amps":z_amps_array, "time": evo_time }
                )
        return dataset 

def CZ_couplerz(z_amps_range,z_amps_resolution,couplerz_amps_range,couplerz_amps_resolution,ro_element,flux_Qi,excited_Qi,flux_Ci,preprocess,qmm,config,n_avg=100,initializer=None,simulate=True):
    """
    find the point of turn off qubit-qubit coupling with coupler\n
    z pulse time is fixed at 100 ns
    """
    z_amps_array = np.arange(z_amps_range[0],z_amps_range[1],z_amps_resolution)
    couplerz_amps_array = np.arange(couplerz_amps_range[0],couplerz_amps_range[1],couplerz_amps_resolution)
    amps_len = len(z_amps_array)
    camps_len = len(couplerz_amps_array)

    with program() as cz:
        iqdata_stream = multiRO_declare( ro_element )
        n = declare(int)
        n_st = declare_stream()
        couplerz_amps = declare(fixed)  
        z_amps = declare(fixed)
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(couplerz_amps, couplerz_amps_array)):
                with for_(*from_array(z_amps,z_amps_array )):
                    # initializaion
                    if initializer is None:
                        wait(1*u.us,ro_element)
                    else:
                        try:
                            initializer[0](*initializer[1])
                        except:
                            wait(1*u.us,ro_element)
                    
                    # operation
                    if excited_Qi != []: 
                        for excited_Qi in excited_Qi:
                            play("x180", f"q{excited_Qi}_xy")
                    align()
                    wait(20 *u.ns)
                    play("const" * amp(z_amps), f"q{flux_Qi}_z", duration=35)
                    play("const" * amp(couplerz_amps), f"q{flux_Ci}_z", duration=35)
                    align()               
                    wait(20 * u.ns)
                    play( "const"*amp( -0.4 ), "q7_z", duration=100)
                    multiRO_measurement(iqdata_stream, ro_element, weights="rotated_") 
            save(n, n_st)

        with stream_processing():
            n_st.save("iteration")
            match preprocess:
                case "shot":
                    multiRO_pre_save(iqdata_stream, ro_element, (n_avg,camps_len,amps_len) ,stream_preprocess="shot")
                case _:
                    multiRO_pre_save(iqdata_stream, ro_element, (camps_len,amps_len))
    if simulate:
        simulation_config = SimulationConfig(duration=20000)  # In clock cycles = 4ns
        job = qmm.simulate(config, cz, simulation_config)
        job.get_simulated_samples().con1.plot()
        #job.get_simulated_samples().con2.plot()
        plt.show()
    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(cz)
        # Get results from QUA program
        ro_ch_name = []
        for r_name in ro_element:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")
        data_list = ro_ch_name + ["iteration"]   
        results = fetching_tool(job, data_list=data_list, mode="live")

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

        match preprocess:
            case "shot":
                for r_idx, r_name in enumerate(ro_element):
                    output_data[r_name] = ( ["mixer","shot","c_amps","amps"],
                                np.squeeze(np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) ))
                dataset = xr.Dataset(
                    output_data,
                    coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(n_avg),"c_amps":couplerz_amps_array, "amps": z_amps_array }
                )
            case _:
                for r_idx, r_name in enumerate(ro_element):
                    output_data[r_name] = ( ["mixer","c_amps","amps"],
                                np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) )
                dataset = xr.Dataset(
                    output_data,
                    coords={ "mixer":np.array(["I","Q"]), "c_amps": couplerz_amps_array, "amps": z_amps_array }
                )

        return dataset
    
def CZ_ramsey(time_max,time_resolution,couplerz_amps_range,couplerz_amps_resolution,z_amps,ro_element,flux_Qi,excited_Qi,flux_Ci,preprocess,qmm,config,n_avg=100,initializer=None,simulate=True):
    cc_resolution = (time_resolution/4.)*u.us
    cc_max_qua = (time_max/4.)*u.us
    cc_qua = np.arange( 4, cc_max_qua, cc_resolution)
    print(cc_qua)
    evo_time = cc_qua*4
    
    couplerz_amps_array = np.arange(couplerz_amps_range[0],couplerz_amps_range[1],couplerz_amps_resolution)
    amp_len = len(couplerz_amps_array)
    time_len = len(evo_time)

    with program() as cz:
        iqdata_stream = multiRO_declare( ro_element )
        n = declare(int)
        n_st = declare_stream()
        cc = declare(int)  
        couplerz_amps = declare(fixed)
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(cc,cc_qua )):
                with for_(*from_array(couplerz_amps, couplerz_amps_array)):
                    # initializaion
                    if initializer is None:
                        wait(1*u.us,ro_element)
                    else:
                        try:
                            initializer[0](*initializer[1])
                        except:
                            wait(1*u.us,ro_element)
                    
                    # operation
                    if excited_Qi != []: 
                        for excited_Qi in excited_Qi:
                            play("x180", f"q{excited_Qi}_xy")
                    align()
                    wait(20 *u.ns)
                    play("const" * amp(z_amps), f"q{flux_Qi}_z", duration=cc)
                    play("const" * amp(couplerz_amps),f"q{flux_Ci}_z",duration=cc) 
                    wait(20 *u.ns)
                    play( "const"*amp( -0.4 ), "q7_z", duration=100)
                    wait(cc)
                    multiRO_measurement(iqdata_stream, ro_element, weights="rotated_") 
            save(n, n_st)

        with stream_processing():
            n_st.save("iteration")
            match preprocess:
                case "shot":
                    multiRO_pre_save(iqdata_stream, ro_element, (n_avg,time_len,amp_len) ,stream_preprocess="shot")
                case _:
                    multiRO_pre_save(iqdata_stream, ro_element, (time_len,amp_len))
    if simulate:
        simulation_config = SimulationConfig(duration=20000)  # In clock cycles = 4ns
        job = qmm.simulate(config, cz, simulation_config)
        job.get_simulated_samples().con1.plot()
        #job.get_simulated_samples().con2.plot()
        plt.show()
    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(cz)
        # Get results from QUA program
        ro_ch_name = []
        for r_name in ro_element:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")
        data_list = ro_ch_name + ["iteration"]   
        results = fetching_tool(job, data_list=data_list, mode="live")

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
        match preprocess:
            case "shot":
                for r_idx, r_name in enumerate(ro_element):
                    output_data[r_name] = ( ["mixer","shot","time","c_amps"],
                                np.squeeze(np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) ))
                dataset = xr.Dataset(
                    output_data,
                    coords={ "mixer":np.array(["I","Q"]), "shot":np.arange(n_avg), "time": evo_time, "c_amps":couplerz_amps_array}
                )
            case _:
                for r_idx, r_name in enumerate(ro_element):
                    output_data[r_name] = ( ["mixer","time","c_amps"],
                                np.array([fetch_data[r_idx*2], fetch_data[r_idx*2+1]]) )
                dataset = xr.Dataset(
                    output_data,
                    coords={ "mixer":np.array(["I","Q"]), "time": evo_time, "c_amps":couplerz_amps_array}
                )
        return dataset 
    
def plot_cz_chavron(dataset,ro_name,chavron_type):
    """
    x in shape (N,) \n
    y in shape (M,) \n
    z in shape (M,N) \n
    N is pulse time \n
    M is flux
    """
    fig, ax = plt.subplots()
    if chavron_type == "qubit-time":
        flux = dataset.coords["amps"].values
        time = dataset.coords["time"].values
        ax.pcolormesh(time,flux,dataset[ro_name].values[0],cmap='RdBu')
        ax.set_xlabel("interaction time (ns)")
        ax.set_ylabel("qubit flux (V)")
    elif chavron_type == "coupler-qubit":
        coupler_flux = dataset.coords["c_amps"].values
        flux = dataset.coords["amps"].values
        ax.pcolormesh(flux,coupler_flux,dataset[ro_name].values[0],cmap='RdBu')
        ax.set_xlabel("qubit flux (V)")
        ax.set_ylabel("coupler flux (V)")
    else:
        time = dataset.coords["time"].values
        coupler_flux = dataset.coords["c_amps"].values
        ax.pcolormesh(coupler_flux,time,dataset[ro_name].values[0],cmap='RdBu')
        ax.set_xlabel("coupler flux (V)")
        ax.set_ylabel("interaction time (ns)")
        ax.axvline(x=0.0, color='black', linestyle='--', label='bias point')
        ax.legend()
    ax.set_title('CZ chavron')
    #a1 = ax.pcolormesh(x,y,z,cmap='RdBu')
    #plt.colorbar(a1,ax=ax, label="|11> population")
    return fig

def plot_coupler_z_vs_time(x,y,z,ax=None):
    """
    x in shape (N,) \n
    y in shape (M,) \n
    z in shape (M,N) \n
    N is pulse time \n
    M is flux
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    # Set the title and labels
    ax.set_title('pcolormesh')
    ax.set_xlabel("coupler flux (V)")  # Reversed
    ax.set_ylabel("interaction time (ns)")  # Reversed
    
    # Reverse x and y in pcolormesh
    a1 = ax.pcolormesh(y, x, z.T, cmap='RdBu')
    
    # Add colorbar
    plt.colorbar(a1, ax=ax, label="|11> population")

from qualang_tools.plot.fitting import Fit
import numpy as np

def plot_cz_frequency_vs_flux(x, y, z, threshold=5, ax=None):
    """
    x in shape (N,) \n
    y in shape (M,) \n
    z in shape (M,N) \n
    N is pulse time \n
    M is flux
    threshold: maximum allowable difference between adjacent points (in MHz)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Arrays to store fitted frequencies
    freqs = []

    # Perform Ramsey fit for each coupler flux (y-axis)
    for i in range(len(y)):
        try:
            # Try to perform Ramsey fit on the interaction time (x) for each coupler flux
            fit = Fit()
            ana_dict_pos = fit.ramsey(x, z[i], plot=False)  # Perform the Ramsey fit
            freq_pos = ana_dict_pos['f'][0] * 1e3  # Frequency in MHz
            freqs.append(freq_pos)  # Store frequency
        except Exception as e:
            # If fit fails, append NaN to mark this point as invalid
            freqs.append(np.nan)

    # Convert lists to arrays for further processing
    freqs = np.array(freqs)

    # Remove frequency jumps that are too large
    for i in range(1, len(freqs) - 1):
        if (abs(freqs[i] - freqs[i - 1]) > threshold and 
            abs(freqs[i] - freqs[i + 1]) > threshold):
            freqs[i] = np.nan  # Mark this point as invalid if it jumps too much

    # Plot frequency vs coupler flux
    ax.set_title('Frequency vs Coupler Flux')
    ax.set_xlabel("Coupler flux (V)")
    ax.set_ylabel("Frequency (MHz)")

    # Plotting frequency with missing (NaN) points automatically skipped
    ax.plot(y, freqs, 'o', label="Fitted Frequency (MHz)")
    ax.legend()

    # Optionally return the fitted frequencies if needed
    return freqs

def plot_cz_period_vs_flux(x, y, z, threshold=5, ax=None):
    """
    x in shape (N,) \n
    y in shape (M,) \n
    z in shape (M,N) \n
    N is pulse time \n
    M is flux
    threshold: maximum allowable difference between adjacent points (in MHz)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Arrays to store fitted periods (calculated as 1/frequency)
    periods = []

    # Perform Ramsey fit for each coupler flux (y-axis)
    for i in range(len(y)):
        try:
            # Try to perform Ramsey fit on the interaction time (x) for each coupler flux
            fit = Fit()
            ana_dict_pos = fit.ramsey(x, z[i], plot=False)  # Perform the Ramsey fit
            freq_pos = ana_dict_pos['f'][0] * 1e3  # Frequency in MHz
            # Convert frequency to period (T = 1/f) in nanoseconds
            if freq_pos > 0:  # Ensure frequency is positive to avoid division by zero
                period = 1 / freq_pos*1e3
                periods.append(period)  # Store period
            else:
                periods.append(np.nan)  # Handle zero or negative frequencies
        except Exception as e:
            # If fit fails, append NaN to mark this point as invalid
            periods.append(np.nan)

    # Convert lists to arrays for further processing
    periods = np.array(periods)

    # Remove period jumps that are too large (using threshold in frequency domain)
    for i in range(1, len(periods) - 1):
        if (abs(periods[i] - periods[i - 1]) > threshold and 
            abs(periods[i] - periods[i + 1]) > threshold):
            periods[i] = np.nan  # Mark this point as invalid if it jumps too much

    # Plot period vs coupler flux
    ax.set_title('Period vs Coupler Flux')
    ax.set_xlabel("Coupler flux (V)")
    ax.set_ylabel("Period (ns)")

    # Plotting periods with missing (NaN) points automatically skipped
    ax.plot(y, periods, 'o', linestyle='', label="Fitted Period (ns)")
    ax.legend()

    # Optionally return the fitted periods if needed
    return periods


def plot_cz_couplerz(x,y,z,ax=None):
    """
    x in shape (N,) \n
    y in shape (M,) \n
    z in shape (M,N) \n
    N is qubit flux \n
    M is coupler flux
    """
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_title('pcolormesh')
    ax.set_xlabel("qubit flux (V)")
    ax.set_ylabel("coupler flux (V)")
    a1 = ax.pcolormesh(x,y,z,cmap='RdBu')
    plt.colorbar(a1,ax=ax, label="|11> population")

def plot_cz_ramsey(x,y,z,ax=None):
    """
    x in shape (N,) \n
    y in shape (M,) \n
    z in shape (M,N) \n
    N is coupler flux \n
    M is time
    """
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_title('pcolormesh')
    ax.set_xlabel("coupler flux (V)")
    ax.set_ylabel("time (ns)")
    ax.axvline(x=0.0, color='black', linestyle='--', label='bias point')
    a1 = ax.pcolormesh(x,y,z,cmap='RdBu')
    ax.legend()
    plt.colorbar(a1,ax=ax, label="|11> population")

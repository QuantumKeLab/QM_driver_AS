from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from exp.RO_macros import multiRO_declare, multiRO_measurement, multiRO_pre_save
import warnings
warnings.filterwarnings("ignore")
import exp.config_par as gc
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
import xarray as xr
import numpy as np
from exp.QMMeasurement import QMMeasurement

class ZZCouplerFreqRamsey( QMMeasurement ):

    """
    Parameters:
    Search ZZ free point with the given Z flux(voltage) range.\n

    virtual detune:\n
    is a float, Unit in MHz. To get real detune. \n
    flux_range:\n
    unit in mV.\n
    resolution:\n
    unit in mV.\n
    target_element: ["q0_ro"], temporarily support only 1 element in the list.\n
    who do ramsey.\n
    crosstalk_element: ["q2_ro"], temporarily support only 1 element in the list.\n
    who is applied X or I to measure ZZ (not Z line) crosstalk is.\n
    coupler_element: ["q1_ro"], temporarily support only 1 element in the list.\n
    the coupler between target_element and crosstalk_element, whose frequency is tuned to find ZZ free point. \n
    Return: \n

    """
    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q0_ro"]
        self.zz_detector_xy = ["q0_xy"]
        self.zz_source_xy = ["q1_xy"]
        self.coupler_z = ["q2_z"]
        self.initializer = None
        
        self.virtual_detune = 5
        self.flux_range = ( -0.1, 0.1 )
        self.resolution = 0.001

    def _get_qua_program( self ):
        
        self.virtual_detune_qua = np.array([self.virtual_detune*u.MHz, -self.virtual_detune*u.MHz])
        self.flux_qua = self._lin_flux_array( )
        self.evo_time_tick_qua = self._evo_time_tick_array( )
        with program() as ZZfree:
            iqdata_stream = multiRO_declare( self.ro_elements )
            n = declare(int)
            n_st = declare_stream()

            t = declare(int)  # QUA variable for the idle time, unit in clock cycle
            phi = declare(fixed)  # Phase to apply the virtual Z-rotation
            phi_idx = declare(bool,)
            X_idx = declare(bool,)
            dc = declare(fixed) 

            with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                with for_each_( X_idx, [True, False]):
                    with for_each_( phi_idx, [True, False]):
                        with for_(*from_array(dc, self.flux_qua)):
                            with for_( *from_array(t, self.evo_time_tick_qua) ):
                                # Initialization
                                # Wait for the resonator to deplete
                                if self.initializer is None:
                                    wait(10 * u.us, self.ro_elements)
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        print("initializer didn't work!")
                                        wait(1 * u.us, self.ro_elements) 
                                # Operation
                                True_value = Cast.mul_fixed_by_int(self.virtual_detune_qua[0] * 1e-9, 4 * t)
                                False_value = Cast.mul_fixed_by_int(self.virtual_detune_qua[1] * 1e-9, 4 * t)
                                assign(phi, Util.cond(phi_idx, True_value, False_value))

                                with if_(X_idx):
                                    play("x180", self.zz_source_xy[0])   # conditional x180 gate
                                    align()

                                play("x90", self.zz_detector_xy[0])  # 1st x90 gate
                                align()
                                play("const"*amp(dc), self.coupler_z[0], t)    # const 預設0.1
                                wait(t, self.zz_detector_xy[0])
                                align()
                                frame_rotation_2pi(phi, self.zz_detector_xy[0])  # Virtual Z-rotation
                                play("x90", self.zz_detector_xy[0])  # 2st x90 gate
                                align()
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                multiRO_pre_save( iqdata_stream, self.ro_elements, (2, 2, len(self.flux_qua), len(self.evo_time_tick_qua)))
                n_st.save("iteration")

        return ZZfree

    def _get_fetch_data_list( self ):
        ro_ch_name = []
        for r_name in self.ro_elements:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")

        data_list = ro_ch_name + ["iteration"]   
        return data_list
     
    def _data_formation( self ):
        output_data = {}

        for r_idx, r_name in enumerate(self.ro_elements):
            output_data[r_name] = ( ["mixer","X","virtual_detune","flux","time"],
                                np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
        dataset = xr.Dataset(
            output_data,
            coords={"mixer":np.array(["I","Q"]), "X": np.array([True, False]), "virtual_detune": self.virtual_detune_qua, "flux": self.flux_qua, "time": 4*self.evo_time_tick_qua}
        )
        return dataset
     
    def _lin_flux_array( self ):
        return np.arange( self.flux_range[0], self.flux_range[1], self.resolution )
    
    def _evo_time_tick_array( self ):
        point_per_period = 20
        Ramsey_period = (1e3/self.virtual_detune)* u.ns
        tick_resolution = (Ramsey_period//(4*point_per_period))
        evo_time_tick_max = tick_resolution *point_per_period*6
        evo_time_tick = np.arange( 4, evo_time_tick_max, tick_resolution)
        return evo_time_tick
    
class ZZCouplerFreqEcho( QMMeasurement ):

    """
    Parameters:
    Search ZZ free point with the given Z flux(voltage) range.\n

    flux_range:\n
    unit in mV.\n

    resolution:\n
    unit in mV.\n

    ro_elements: ["q0_ro"], temporarily support only 1 element in the list.\n
    who do ramsey.\n
    zz_source_xy: ["q1_xy"], temporarily support only 1 element in the list.\n
    who is applied X or I to measure ZZ (not Z line) crosstalk is.\n
    coupler_z: ["q2_z"], temporarily support only 1 element in the list.\n
    the coupler between target_element and crosstalk_element, whose frequency is tuned to find ZZ free point. \n
    Return: \n

    """
    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q1_ro"]
        self.zz_detector_xy = ["q1_xy"]
        self.zz_source_xy = ["q2_xy"]
        self.coupler_z = ["q6_z"]
        self.predict_detune = 0.1

        self.preprocess = "ave"
        self.initializer = None
        
        self.time_range = (4, 20000) #ns
        self.time_resolution = 200
        self.flux_range = ( -0.1, 0.1 )
        self.resolution = 0.001

    def _get_qua_program( self ):
        self.flux_qua = self._lin_flux_array( )
        # self.evo_time_tick_qua = self._evo_time_tick_array( )
        self.evo_time_tick_qua = np.arange(self.time_range[0], self.time_range[1], self.time_resolution)//4
        with program() as ZZfree:
            iqdata_stream = multiRO_declare( self.ro_elements )
            n = declare(int)
            n_st = declare_stream()

            t = declare(int)  # QUA variable for the idle time, unit in clock cycle
            dc = declare(fixed) 

            with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                with for_(*from_array(dc, self.flux_qua)):
                    with for_( *from_array(t, self.evo_time_tick_qua) ):
                        # Initialization
                        # Wait for the resonator to deplete
                        if self.initializer is None:
                            wait(10 * u.us, self.ro_elements)
                        else:
                            try:
                                self.initializer[0](*self.initializer[1])
                            except:
                                print("initializer didn't work!")
                                wait(1 * u.us, self.ro_elements) 

                        play("x90", self.zz_detector_xy[0])  # 1st x90 gate
                        align()
                        wait(5)
                        play("const"*amp(dc), self.coupler_z[0], t)    # const 預設0.5
                        align()
                        wait(5)
                        play("x180", self.zz_detector_xy[0])     #flip
                        play("x180", self.zz_source_xy[0])      #make ZZ crosstalk
                        align()
                        wait(5)


                        play("const"*amp(dc), self.coupler_z[0], t)    # const 預設0.5
                        align()
                        wait(5)
                        # frame_rotation_2pi(0.5, self.zz_detector_xy[0])  # Virtual Z-rotation
                        play("-x90", self.zz_detector_xy[0])  # 2nd x90 gate
                        align()
                        
                        # Readout
                        multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.flux_qua), len(self.evo_time_tick_qua)))
                n_st.save("iteration")

        return ZZfree

    def _get_fetch_data_list( self ):
        ro_ch_name = []
        for r_name in self.ro_elements:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")

        data_list = ro_ch_name + ["iteration"]   
        return data_list
     

    def _data_formation( self ):
        coords = { 
            "mixer":np.array(["I","Q"]), 
            "flux":self.flux_qua * gc.get_const_wf(self.coupler_z[0],self.config),
            "time":4*self.evo_time_tick_qua,
            #"prepare_state": np.array([0,1])
            }
        match self.preprocess:
            case "shot":
                dims_order = ["mixer","shot", "flux", "time"]
                coords["shot"] = np.arange(self.shot_num)
            case _:
                dims_order = ["mixer","flux","time"]

        output_data = {}
        for r_idx, r_name in enumerate(self.ro_elements):
            data_array = np.array([ self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]])
            output_data[r_name] = ( dims_order, np.squeeze(data_array))

        dataset = xr.Dataset( output_data, coords=coords )

        # dataset = dataset.transpose("mixer", "prepare_state", "frequency", "amp_ratio")

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
        for xy in self.zz_detector_xy:
            self.ref_xy_IF.append(gc.get_IF(xy, self.config))
            self.ref_xy_LO.append(gc.get_LO(xy, self.config))
        for xy in self.zz_source_xy:
            self.ref_xy_IF.append(gc.get_IF(xy, self.config))
            self.ref_xy_LO.append(gc.get_LO(xy, self.config))

        self.z_offset = []
        self.z_amp = []
        for z in self.coupler_z:
            self.z_offset.append( gc.get_offset(z, self.config ))
            self.z_amp.append(gc.get_const_wf(z, self.config ))
     
    def _lin_flux_array( self ):
        return np.arange( self.flux_range[0], self.flux_range[1], self.resolution )
    
    def _evo_time_tick_array( self ):
        point_per_period = 100
        Ramsey_period = (1e3/self.predict_detune)* u.ns
        tick_resolution = (Ramsey_period//(4*point_per_period))
        evo_time_tick_max = tick_resolution *point_per_period*6
        evo_time_tick = np.arange( 4, evo_time_tick_max, tick_resolution)
        return evo_time_tick
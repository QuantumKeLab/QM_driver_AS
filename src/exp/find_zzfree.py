from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from exp.RO_macros import multiRO_declare, multiRO_measurement, multiRO_pre_save
import warnings
warnings.filterwarnings("ignore")
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
import xarray as xr
import numpy as np
from exp.QMMeasurement import QMMeasurement

class find_ZZfree( QMMeasurement ):

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
        self.target_xy = ["q0_xy"]
        self.crosstalk_xy = ["q1_xy"]
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
            iqdata_stream = multiRO_declare( self.ro_elements[0] )
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
                                    wait(10 * u.us, self.ro_elements[0])
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        print("initializer didn't work!")
                                        wait(1 * u.us, self.ro_elements[0]) 
                                # Operation
                                True_value = Cast.mul_fixed_by_int(self.virtual_detune_qua[0] * 1e-9, 4 * t)
                                False_value = Cast.mul_fixed_by_int(self.virtual_detune_qua[1] * 1e-9, 4 * t)
                                assign(phi, Util.cond(phi_idx, True_value, False_value))

                                with if_(X_idx):
                                    play("x180", self.crosstalk_xy[0])   # conditional x180 gate
                                    align()

                                play("x90", self.target_xy[0])  # 1st x90 gate
                                align()
                                play("const"*amp(dc*2.), self.coupler_z[0], t)    # const 預設0.1
                                wait(t, self.target_xy[0])
                                align()
                                frame_rotation_2pi(phi, self.target_xy[0])  # Virtual Z-rotation
                                play("x90", self.target_xy[0])  # 2st x90 gate
                                align()
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements[0], weights="rotated_") 
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                multiRO_pre_save( iqdata_stream, self.ro_elements[0], (2, 2, len(self.flux_qua), len(self.evo_time_tick_qua)))
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
    
class pure_ZZfree( QMMeasurement ):

    """
    Parameters:
    Search ZZ free point with the given Z flux(voltage) range.\n

    flux_range:\n
    unit in mV.\n

    resolution:\n
    unit in mV.\n

    ro_elements: ["q0_ro"], temporarily support only 1 element in the list.\n
    who do ramsey.\n
    crosstalk_xy: ["q1_xy"], temporarily support only 1 element in the list.\n
    who is applied X or I to measure ZZ (not Z line) crosstalk is.\n
    coupler_z: ["q2_z"], temporarily support only 1 element in the list.\n
    the coupler between target_element and crosstalk_element, whose frequency is tuned to find ZZ free point. \n
    Return: \n

    """
    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q0_ro"]
        self.target_xy = ["q0_xy"]
        self.crosstalk_xy = ["q1_xy"]
        self.coupler_z = ["q2_z"]
        self.initializer = None
        
        self.flux_range = ( -0.1, 0.1 )
        self.resolution = 0.001

    def _get_qua_program( self ):
        self.flux_qua = self._lin_flux_array( )
        self.evo_time_tick_qua = self._evo_time_tick_array( )
        with program() as ZZfree:
            iqdata_stream = multiRO_declare( self.ro_elements[0] )
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
                            wait(10 * u.us, self.ro_elements[0])
                        else:
                            try:
                                self.initializer[0](*self.initializer[1])
                            except:
                                print("initializer didn't work!")
                                wait(1 * u.us, self.ro_elements[0]) 

                        play("x90", self.target_xy[0])  # 1st x90 gate
                        align()
                        play("const"*amp(dc*2.), self.coupler_z[0], t)    # const 預設0.5
                        wait(t, self.target_xy[0])
                        align()

                        play("x180", self.target_xy[0])     #flip
                        play("x180", self.crosstalk_xy[0])      #make ZZ crosstalk
                        align()

                        play("const"*amp(dc*2.), self.coupler_z[0], t)    # const 預設0.5
                        wait(t, self.target_xy[0])
                        align()
                        play("x90", self.target_xy[0])  # 2nd x90 gate
                        align()
                        
                        # Readout
                        multiRO_measurement( iqdata_stream, self.ro_elements[0], weights="rotated_") 
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                multiRO_pre_save( iqdata_stream, self.ro_elements[0], (len(self.flux_qua), len(self.evo_time_tick_qua)))
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
            output_data[r_name] = ( ["mixer","flux","time"],
                                np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
        dataset = xr.Dataset(
            output_data,
            coords={"mixer":np.array(["I","Q"]), "flux": self.flux_qua, "time": 4*self.evo_time_tick_qua}
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
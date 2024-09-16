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

warnings.filterwarnings("ignore")
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
from exp.config_par import get_offset

class FluxCrosstalk( QMMeasurement ):

    """
    Parameters:
    measure flux crosstalk with the given Z flux(voltage) range.\n

    expect_crosstalk:\n
    no unit.\n

    z_modify_range:\n
    unit in V.\n

    z_resolution:\n
    How many points is z_modify_range divided into.\n

    z_time:\n
    unit in micro second.\n

    measure_method:\n
    ramsey or long drive.\n
    see https://docs.google.com/presentation/d/18QqU46g3KktW5dPgIXpiLrI3leNX1FnE/edit?usp=drive_link&ouid=115212552955966668572&rtpof=true&sd=true\n
    page 2\n

    z_method:\n
    pulse or offset.\n
    The offset mode does not apply corrections from the crosstalk matrix.\n
    Only pulse mode does.\n
    2024/07/22\n

    ro_elements:\n
    who read out.\n

    detector_qubit:\n
    who is driven.\n

    detector_is_coupler:\n
    If detector is coupler, readout is not so easy.\n
    I haven't do this.\n

    crosstalk_qubit:\n
    Who provide its z_line to make flux crosstalk to others.\n 
    """

    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q8_ro"]
        self.detector_qubit = "q8"
        self.detector_is_coupler = False
        self.crosstalk_qubit = "q4"
        self.initializer = None

        self.expect_crosstalk = 0.5
        self.detector_bias = -0.
        self.z_modify_range = 0.4
        self.z_resolution = 0.016
        self.z_time = 0.16
        self.z_time_qua = self.z_time/4 *u.us

        self.measure_method = "long_drive"   #long_drive, ramsey
        self.z_method = "pulse"     #offset, pulse


    def _get_qua_program( self ):
        self.crosstalk_z_qua = self._lin_z_array( )
        self.detector_z_qua = self.expect_crosstalk * self.crosstalk_z_qua + self.detector_bias
        self.z_time_cc = self.z_time*u.us//4
        self.pi_length = self._get_pi_length( )
        self.pi_amp_ratio = self.pi_length/(self.z_time*u.us)
        if(self.detector_is_coupler == False):
            if ((self.measure_method == "ramsey") & (self.z_method == "pulse")):
                with program() as Ramsey_z_pulse:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5)
                                play("const"*amp(z_crosstalk*2.), f"{self.crosstalk_qubit}_z", self.z_time_cc)         #const 預設0.5, microsecond transfrom to cc
                                play("const"*amp(z_detector*2.), f"{self.detector_qubit}_z", self.z_time_cc)
                                align()
                                wait(5) 
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5)  
                                
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return Ramsey_z_pulse
            
            elif((self.measure_method == "long_drive") & (self.z_method == "pulse")):
                print("有打這個")
                with program() as LongDrive_z_pulse:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                play("const"*amp(z_crosstalk*10.), f"{self.crosstalk_qubit}_z", self.z_time_cc+10)         #const 預設0.5
                                play("const"*amp(z_detector*10.), f"{self.detector_qubit}_z", self.z_time_cc+10)
                                wait(5)
                                play( "x180"*amp(self.pi_amp_ratio), f"{self.detector_qubit}_xy", duration=self.z_time_cc )
                                # play("const"*amp( 0.2 ), f"{self.detector_qubit}_xy", duration=self.z_time_cc)
                                # wait(17-5, f"{self.crosstalk_qubit}_z")    #不加這個wait的話x180會比z慢17cc，到底為啥???
                                # wait(17-5, f"{self.detector_qubit}_z")    #不加這個wait的話x180會比z慢17cc，到底為啥???

                                wait(self.z_time_cc+10)

                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return LongDrive_z_pulse
            
            elif((self.measure_method == "ramsey") & (self.z_method == "offset")):
                with program() as Ramsey_z_offset:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5)
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config)+z_crosstalk )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config)+z_detector )
                                wait(self.z_time_cc) 
                                align()
                                wait(5) 
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config) )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config) )
                                align()
                                wait(5)
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5) 
                                
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return Ramsey_z_offset

            elif((self.measure_method == "long_drive") & (self.z_method == "offset")):
                with program() as LongDrive_z_offset:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
                                # Initialization
                                # Wait for the resonator to deplete
                                # set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config)+z_crosstalk )
                                # set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config)+z_detector )
                                if self.initializer is None:
                                    wait(10 * u.us, self.ro_elements)
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        print("initializer didn't work!")
                                        wait(1 * u.us, self.ro_elements) 

                                # Opration
                                play( "x180"*amp(self.pi_amp_ratio), f"{self.detector_qubit}_xy", duration=self.z_time_cc )
                                wait(22-5)  #不加這個wait的話x180會比z慢22cc，到底為啥???
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config)+z_crosstalk )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config)+z_detector )
                                wait(self.z_time_cc+10)
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config) )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config) )
                                
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return LongDrive_z_offset

            else:
                print("unknown measure_method or z_method")
        else:
            with program() as Is_coupler:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                # play( "x180", f"{self.detector_qubit}_xy", duration=self.z_time_cc )
                                # play("const"*amp( 0.05 ), f"{self.detector_qubit}_xy", duration=self.z_time_cc)
                                # wait(17-5)    #不加這個wait的話x180會比z慢17cc，到底為啥???
                                play("const"*amp(z_crosstalk*2.), f"{self.crosstalk_qubit}_z", self.z_time_cc+10)         #const 預設0.5
                                play("const"*amp(z_detector*2.), f"{self.detector_qubit}_z", self.z_time_cc+10)
                                wait(5)
                                play("const"*amp( 0.2 ), f"{self.detector_qubit}_xy", duration=self.z_time_cc)
                                for i, ro in enumerate(self.ro_elements):
                                    # wait(4000, self.ro_elements)
                                    wait( int(self.z_time_qua-gc.get_ro_length(ro,self.config)//4), ro )

                                # wait(self.z_time_cc+15)

                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

            return Is_coupler



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
            output_data[r_name] = ( ["mixer", "crosstalk_z", "detector_z"],
                                np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
        dataset = xr.Dataset(
            output_data,
            coords={"mixer":np.array(["I","Q"]), "crosstalk_z": self.crosstalk_z_qua, "detector_z": self.detector_z_qua}
        )

        self._attribute_config()
        dataset.attrs["expect_crosstalk"] = self.expect_crosstalk
        dataset.attrs["z_time"] = self.z_time
        dataset.attrs["detector_bias"] = self.detector_bias
        dataset.attrs["z_modify_range"] = self.z_modify_range
        dataset.attrs["z_resolution"] = self.z_resolution
        dataset.attrs["measure_method"] = self.measure_method
        dataset.attrs["z_method"] = self.z_method
        dataset.attrs["detector_qubit"] = self.detector_qubit
        dataset.attrs["crosstalk_qubit"] = self.crosstalk_qubit

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
        self.ref_xy_IF.append(gc.get_IF(f"{self.detector_qubit}_xy", self.config))
        self.ref_xy_LO.append(gc.get_LO(f"{self.detector_qubit}_xy", self.config))
        self.ref_xy_IF.append(gc.get_IF(f"{self.crosstalk_qubit}_xy", self.config))
        self.ref_xy_LO.append(gc.get_LO(f"{self.crosstalk_qubit}_xy", self.config))


        self.z_offset = []
        self.z_amp = []
        self.z_offset.append( gc.get_offset(f"{self.detector_qubit}_z", self.config ))
        self.z_amp.append(gc.get_const_wf(f"{self.detector_qubit}_z", self.config ))
        self.z_offset.append( gc.get_offset(f"{self.crosstalk_qubit}_z", self.config ))
        self.z_amp.append(gc.get_const_wf(f"{self.crosstalk_qubit}_z", self.config ))

 
    def _lin_z_array( self ):
        return np.arange( -self.z_modify_range, self.z_modify_range, self.z_resolution )

    def _get_pi_length( self ):
        pi_length = self.config["pulses"][f"{self.detector_qubit}_xy_x180_pulse"]["length"]
        return pi_length

class FluxCrosstalk_2points( QMMeasurement ):

    """
    Parameters:
    measure flux crosstalk with the given Z flux(voltage) range.\n

    expect_crosstalk:\n
    no unit.\n

    z_modify_range:\n
    unit in V.\n

    z_resolution:\n
    How many points is z_modify_range divided into.\n

    z_time:\n
    unit in micro second.\n

    measure_method:\n
    ramsey or long drive.\n
    see https://docs.google.com/presentation/d/18QqU46g3KktW5dPgIXpiLrI3leNX1FnE/edit?usp=drive_link&ouid=115212552955966668572&rtpof=true&sd=true\n
    page 2\n

    z_method:\n
    pulse or offset.\n
    The offset mode does not apply corrections from the crosstalk matrix.\n
    Only pulse mode does.\n
    2024/07/22\n

    ro_elements:\n
    who read out.\n

    detector_qubit:\n
    who is driven.\n

    detector_is_coupler:\n
    If detector is coupler, readout is not so easy.\n
    I haven't do this.\n

    crosstalk_qubit:\n
    Who provide its z_line to make flux crosstalk to others.\n 
    """

    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q8_ro"]
        self.detector_qubit = "q8"
        self.detector_is_coupler = "True"
        self.crosstalk_qubit = "q3"
        self.initializer = None

        self.expect_crosstalk = 0.01
        self.z_modify_range = 0.4
        self.z_resolution = 0.008
        self.z_time = 20
        self.z_time_qua = self.z_time/4 *u.us

        self.measure_method = "long_drive"   #long_drive, ramsey
        self.z_method = "offset"     #offset, pulse


    def _get_qua_program( self ):
        self.crosstalk_z_qua = self._lin_z_array( )
        self.detector_z_qua = [self.expect_crosstalk[0], self.expect_crosstalk[-1]] *self.crosstalk_z_qua
        self.z_time_cc = self.z_time*u.us//4
        self.pi_length = self._get_pi_length( )
        self.pi_amp_ratio = self.pi_length/(self.z_time*u.us)
        if(self.detector_is_coupler == False):
            if ((self.measure_method == "ramsey") & (self.z_method == "pulse")):
                with program() as Ramsey_z_pulse:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5)
                                play("const"*amp(z_crosstalk*2.), f"{self.crosstalk_qubit}_z", self.z_time_cc)         #const 預設0.5, microsecond transfrom to cc
                                play("const"*amp(z_detector*2.), f"{self.detector_qubit}_z", self.z_time_cc)
                                align()
                                wait(5) 
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5)  
                                
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return Ramsey_z_pulse
            
            elif((self.measure_method == "long_drive") & (self.z_method == "pulse")):
                with program() as LongDrive_z_pulse:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                play( "x180"*amp(self.pi_amp_ratio), f"{self.detector_qubit}_xy", duration=self.z_time_cc )
                                wait(17-5)    #不加這個wait的話x180會比z慢17cc，到底為啥???
                                play("const"*amp(z_crosstalk*2.), f"{self.crosstalk_qubit}_z", self.z_time_cc+10)         #const 預設0.5
                                play("const"*amp(z_detector*2.), f"{self.detector_qubit}_z", self.z_time_cc+10)
                                wait(self.z_time_cc+15)

                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return LongDrive_z_pulse
            
            elif((self.measure_method == "ramsey") & (self.z_method == "offset")):
                with program() as Ramsey_z_offset:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5)
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config)+z_crosstalk )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config)+z_detector )
                                wait(self.z_time_cc) 
                                align()
                                wait(5) 
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config) )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config) )
                                align()
                                wait(5)
                                play( "x90", f"{self.detector_qubit}_xy" )
                                align()
                                wait(5) 
                                
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return Ramsey_z_offset

            elif((self.measure_method == "long_drive") & (self.z_method == "offset")):
                with program() as LongDrive_z_offset:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
                                # Initialization
                                # Wait for the resonator to deplete
                                # set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config)+z_crosstalk )
                                # set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config)+z_detector )
                                if self.initializer is None:
                                    wait(10 * u.us, self.ro_elements)
                                else:
                                    try:
                                        self.initializer[0](*self.initializer[1])
                                    except:
                                        print("initializer didn't work!")
                                        wait(1 * u.us, self.ro_elements) 

                                # Opration
                                play( "x180"*amp(self.pi_amp_ratio), f"{self.detector_qubit}_xy", duration=self.z_time_cc )
                                wait(22-5)  #不加這個wait的話x180會比z慢22cc，到底為啥???
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config)+z_crosstalk )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config)+z_detector )
                                wait(self.z_time_cc+10)
                                set_dc_offset( f"{self.crosstalk_qubit}_z", "single", get_offset(f"{self.crosstalk_qubit}_z",self.config) )
                                set_dc_offset( f"{self.detector_qubit}_z", "single", get_offset(f"{self.detector_qubit}_z",self.config) )
                                
                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

                return LongDrive_z_offset

            else:
                print("unknown measure_method or z_method")
        else:
            with program() as Is_coupler:
                    iqdata_stream = multiRO_declare( self.ro_elements )
                    n = declare(int)
                    n_st = declare_stream()

                    z_detector = declare(fixed) 
                    z_crosstalk = declare(fixed) 


                    with for_(n, 0, n < self.shot_num, n + 1):  # QUA for_ loop for averaging
                        with for_(*from_array(z_crosstalk, self.crosstalk_z_qua)):
                            with for_( *from_array(z_detector, self.detector_z_qua) ):
                            
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

                                # Opration
                                # play( "x180", f"{self.detector_qubit}_xy", duration=self.z_time_cc )
                                # play("const"*amp( 0.05 ), f"{self.detector_qubit}_xy", duration=self.z_time_cc)
                                # wait(17-5)    #不加這個wait的話x180會比z慢17cc，到底為啥???
                                play("const"*amp(z_crosstalk*2.), f"{self.crosstalk_qubit}_z", self.z_time_cc+10)         #const 預設0.5
                                play("const"*amp(z_detector*2.), f"{self.detector_qubit}_z", self.z_time_cc+10)
                                wait(5)
                                play("const"*amp( 0.05 ), f"{self.detector_qubit}_xy", duration=self.z_time_cc)
                                for i, ro in enumerate(self.ro_elements):
                                    # wait(4000, self.ro_elements)
                                    wait( int(self.z_time_qua-gc.get_ro_length(ro,self.config)//4), ro )

                                # wait(self.z_time_cc+15)

                                # Readout
                                multiRO_measurement( iqdata_stream, self.ro_elements, weights="rotated_") 
                        # Save the averaging iteration to get the progress bar
                        save(n, n_st)

                    with stream_processing():
                        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                        multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.crosstalk_z_qua), len(self.detector_z_qua)))
                        n_st.save("iteration")

            return Is_coupler



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
            output_data[r_name] = ( ["mixer", "crosstalk_z", "detector_z"],
                                np.array([self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]]) )
        dataset = xr.Dataset(
            output_data,
            coords={"mixer":np.array(["I","Q"]), "crosstalk_z": self.crosstalk_z_qua, "detector_z": self.detector_z_qua}
        )

        self._attribute_config()
        dataset.attrs["expect_crosstalk"] = self.expect_crosstalk
        dataset.attrs["z_time"] = self.z_time
        dataset.attrs["z_modify_range"] = self.z_modify_range
        dataset.attrs["z_resolution"] = self.z_resolution
        dataset.attrs["measure_method"] = self.measure_method
        dataset.attrs["z_method"] = self.z_method
        dataset.attrs["detector_qubit"] = self.detector_qubit
        dataset.attrs["crosstalk_qubit"] = self.crosstalk_qubit

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
        self.ref_xy_IF.append(gc.get_IF(f"{self.detector_qubit}_xy", self.config))
        self.ref_xy_LO.append(gc.get_LO(f"{self.detector_qubit}_xy", self.config))
        self.ref_xy_IF.append(gc.get_IF(f"{self.crosstalk_qubit}_xy", self.config))
        self.ref_xy_LO.append(gc.get_LO(f"{self.crosstalk_qubit}_xy", self.config))


        self.z_offset = []
        self.z_amp = []
        self.z_offset.append( gc.get_offset(f"{self.detector_qubit}_z", self.config ))
        self.z_amp.append(gc.get_const_wf(f"{self.detector_qubit}_z", self.config ))
        self.z_offset.append( gc.get_offset(f"{self.crosstalk_qubit}_z", self.config ))
        self.z_amp.append(gc.get_const_wf(f"{self.crosstalk_qubit}_z", self.config ))

 
    def _lin_z_array( self ):
        return np.arange( -self.z_modify_range, self.z_modify_range, self.z_resolution )

    def _get_pi_length( self ):
        pi_length = self.config["pulses"][f"{self.detector_qubit}_xy_x180_pulse"]["length"]
        return pi_length

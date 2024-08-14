


def get_offset( element, config ):
    port_info = config["elements"][element]["singleInput"]["port"]
    con_name = port_info[0]
    port_name = port_info[1]

    offset = config["controllers"][con_name]["analog_outputs"][port_name]["offset"]
    return offset

def get_const_wf( element, config ):
    pulse_name = config["elements"][element]["operations"]["const"]
    waveform_name = config["pulses"][pulse_name]["waveforms"]["single"]
    const_amp = config["waveforms"][waveform_name]["sample"]

    return const_amp

def get_IF( element, config ):
    return config["elements"][element]["intermediate_frequency"]

def get_LO( element, config ):
    return config["elements"][element]["mixInputs"]["lo_frequency"]

def get_ro_length( element, config ):

    pulse_name = config["elements"][element]["operations"]["readout"]
    return config["pulses"][pulse_name]["length"]

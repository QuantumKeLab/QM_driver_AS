
from pathlib import Path
link_path = Path(__file__).resolve().parent/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config
config_obj, spec = import_config( link_path )

from qspec.update import update_z_offset, update_z_crosstalk, update_zWaveform

# [[ 0.99954093  0.00685605 -0.01088031 -0.01185879 -0.01199802]
#  [ 0.02307034  0.99941776 -0.01729353 -0.01564188 -0.01406762]
#  [ 0.01713722  0.01930098  0.99897034 -0.02075049 -0.01185133]
#  [ 0.02022993  0.01678385  0.01898975  0.99923273 -0.02882381]
#  [ 0.01681398  0.01113205  0.00966305 -0.00497131  0.99965561]]

z_infos = {

# e.g.
# "q0":{
#         "offset": 0.0,      # float in volt
#         "crosstalk":{}, 
#         "z_wf":"sin",       # input: 'sin'
#         "z_amp": 0.2,       # float in volt
#         "z_len": 1000,      # int in ns
#         "z_freq": 2.0,      # float in factor
#         "z_phase": 60.0,    # float in degree
#     },

    "q0":{
        "offset": 0.0805,
        "crosstalk":{},
        "z_wf":"sin",
        "z_amp": 0.2,
        "z_len": 1000,
        "z_freq": 2.2,
        "z_phase": 65,
    },
    "q1":{
        "offset": 0.004,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q2":{
        "offset": 0.103,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q3":{
        "offset": 0.095+1.93e-3,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q4":{
        "offset": 0.052,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q5":{
        "offset": 0.0603,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q6":{
        "offset": 0.123,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q7":{
        "offset": 0.0907,#0.097,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
    "q8":{
        "offset": 0.0889,
        "crosstalk":{},
        "z_amp": 0.1,
        "z_len": 1000,
    },
}
updating_qubit = ["q0","q1","q2","q3","q4","q5","q6","q7","q8"]

for i in updating_qubit:
    q_name = i
    wiring = spec.get_spec_forConfig('wire')

    z_wf = z_infos[i]["z_wf"] if "z_wf" in z_infos[i].keys() else "sin"
    z_amp = z_infos[i]["z_amp"] if "z_amp" in z_infos[i].keys() else 0.1
    z_len = z_infos[i]["z_len"] if "z_amp" in z_infos[i].keys() else 40
    z_freq = z_infos[i]["z_freq"] if "z_freq" in z_infos[i].keys() else 1
    z_phase = z_infos[i]["z_phase"] if "z_phase" in z_infos[i].keys() else 0

    # Tune Z offset
    z_info = spec.update_ZInfo_for(target_q=q_name,
                                   offset=z_infos[i]["offset"],
                                   crosstalk=z_infos[i]["crosstalk"],
                                   wf=z_wf,
                                   amp=z_amp,
                                   len=z_len,
                                   freq=z_freq,
                                   phase=z_phase,
                                   )
    # print(wiring[i[0]])
    update_z_offset( config_obj, z_info, wiring[q_name], mode='offset')
    update_z_crosstalk( config_obj, z_info, wiring[q_name])
    update_zWaveform(config_obj, spec.get_spec_forConfig("z"), target_q=q_name )
    config_dict = config_obj.get_config() 


from QM_driver_AS.ultitly.config_io import output_config
output_config( link_path, config_obj, spec )
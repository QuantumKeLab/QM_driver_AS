import numpy as np

from pathlib import Path
link_path = Path(__file__).resolve().parent/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config
config_obj, spec = import_config( link_path )


from config_component.update import update_ReadoutFreqs, update_Readout
new_LO = 6.1
# rin_offset = (+0.01502-0.00016+7.9e-5,+0.01300+1.5e-5) # I,Q
rin_offset = (+0,+0) # I,Q
tof = 280
# init_value of readout amp is 0.2
# ,#
# name, IF, amp, z, len, angle
ro_infos = [
    {
        "name":"q2",  # single resonator
        "IF": -259,
        "amp": 0.03,#*0.3*0.5*4*0.75 *0.8,
        "length":2000,
        "phase": 0
    },
    {
        "name":"q0",
        "IF": -142+0.19,
        "amp": 0.001,#*0.3*0.5*4*0.75 *0.8,
        "length":2000,
        "phase": 0
    },
    {
        "name":"q1",
        "IF": -72.1,
        "amp": 0.001,#*0.3*0.5*4*0.75 *0.8,
        "length":2000,
        "phase": 0
    },
    {    #XXXXXX
        "name":"q3", 
        "IF": 100,
        "amp": 0.025,#*0.3*0.5*4*0.75 *0.8,
        "length":500,
        "phase": 0
    },
]
# cavities = [['q0',+150-33, 0.3*0.1, 0, 2000,0],['q1',+150+0.8, 0.3*0.1*1.5*1.5*1.4, 0.038, 560,84.7],['q8',+150+3, 0.01, 0.1, 2000,0],['q5',+150-36, 0.3*0.05, -0.11, 2000,0]]
for i in ro_infos:
    wiring = spec.get_spec_forConfig('wire')

    f = spec.update_RoInfo_for(target_q=i["name"],LO=new_LO,IF=i["IF"])
    update_ReadoutFreqs(config_obj, f)
    ro_rotated_rad = i["phase"]/180*np.pi
    spec.update_RoInfo_for(i["name"],amp=i["amp"], len=i["length"],rotated=ro_rotated_rad, offset=rin_offset, time=tof)
    update_Readout(config_obj, i["name"], spec.get_spec_forConfig('ro'),wiring)
    # print(spec.get_spec_forConfig('ro')["q1"])

    config_dict = config_obj.get_config() 

from QM_driver_AS.ultitly.config_io import output_config
output_config( link_path, config_obj, spec )

import numpy as np

from pathlib import Path
link_path = Path(__file__).resolve().parent/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config
config_obj, spec = import_config( link_path )


from qspec.update import update_ReadoutFreqs, update_Readout
new_LO = 4.7#5.65
rin_offset = (+0.01502-0.00016+7.9e-5,+0.01300+1.5e-5) # I,Q
# rin_offset = (+0,+0) # I,Q
tof = 280
# init_value of readout amp is 0.2
# ,#
# name, IF, amp, z, len, angle
ro_infos = [
    {    #LO set 5.5
        "name":"q0", 
        "IF": -241.7,
        "amp": 0.3*0.03,
        "length":2000,
        "phase": 0
    },   
    {   # q1-q5 LO set 5.95
        "name":"q1",
        "IF": -132.5+4.44,
        "amp": 0.2*0.02,
        "length":2000,
        "phase": 0
    },
    {   
        "name":"q2",
        "IF": -54.4+9,
        "amp": 0.3*0.01,
        "length":2000,
        "phase": 0
    },

# Not sorted
    
        {
        "name":"q5",
        "IF": -132.5+7.27,
        "amp": 0.2*0.02,#*0.3*0.5*4*0.75 *0.8,
        "length":2000,
        "phase": 0
    },
    {
        "name":"q4",
        "IF": -238.23+5.53,
        "amp": 0.4*0.01,#*0.3*0.5*4*0.75 *0.8,
        "length":2000,
        "phase": 0
    },
    {    
        "name":"q3", 
        "IF": -218.9+2.96,
        "amp": 0.5*0.01,#*0.3*0.5*4*0.75 *0.8,
        "length":2000,
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

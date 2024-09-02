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
    {
        "name":"q0",
        "IF": -360,#-112,
        "amp": 0.2,#0.2,
        "length":2000,
        "phase": 0
    },{
        "name":"q4",
        "IF": 145-0.12,#-112,
        "amp": 0.2*0.5*0.75,
        "length":2000,
        "phase": 0+118.1
    },{
        "name":"q3",
        "IF": -203+1.74-0.27,
        "amp": 0.2*0.75*0.75*0.75*0.75,
        "length":2000,
        "phase": 0
    },{
        "name":"q5",
        "IF": -188,
        "amp": 0.2,
        "length":2000,
        "phase": 0
    },{
        "name":"q1",
        "IF": -160,
        "amp": 0.2,
        "length":2000,
        "phase": 0
    },{
        "name":"q2",
        "IF": -89,
        "amp": 0.2,
        "length":2000,
        "phase": 0
    },
    # {
    #     "name":"q0",
    #     "IF": -360,#-112,
    #     "amp": 0.3*0.2,#0.2,
    #     "length":2000,
    #     "phase": 0
    # },{
    #     "name":"q4",
    #     "IF": -286+350,#-112,
    #     "amp": 0.3*0.2,#0.2,
    #     "length":2000,
    #     "phase": 0
    # },{
    #     "name":"q3",
    #     "IF": -260+350-2.64+0.21,
    #     "amp": 0.3*0.2*2*1.25*1.2,
    #     "length":2000,
    #     "phase": 0+48
    # },{
    #     "name":"q5",
    #     "IF": -188+350-1.38-0.27,
    #     "amp": 0.15*0.06*1.25,#0.15*0.2,
    #     "length":2000,
    #     "phase": 0+146.3
    # },{
    #     "name":"q1",
    #     "IF": -160+350,
    #     "amp": 0.4*0.2,
    #     "length":2000,
    #     "phase": 0
    # },{
    #     "name":"q2",
    #     "IF": -89+350-0.712+1.112,
    #     "amp": 0.3*0.2*1.18,
    #     "length":2000,
    #     "phase": 0+189.1
    # },
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



from pathlib import Path
link_path = Path(__file__).resolve().parent/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config
config_obj, spec = import_config( link_path )

from qspec.update import update_controlFreq, update_controlWaveform

import numpy as np

xy_infos = {
    "q0":{
        "q_freq": 5.263+0.321e-3-0.42e-3-0.253e-3+0.034e-3+0.713e-3-0.018e-3, # GHz
        "LO": 5.3, # GHz
        "pi_amp": 0.2*1.4*1.05*0.9840*0.715,
        "pi_len": 40,
        "90_corr": 1.15*0.9554*0.9651*0.829*1.2
    },
    "q1":{   
        "q_freq": 4.6617-0.412e-3,#4.6648-1.537e-3,#4.740+0.15e-3-2.074e-3+0.101e-3-0.029e-3+0.062e-3-0.062e-3+0.023e-3-0.035e-3, # GHz
        "LO": 4.760, # GHz
        "pi_amp": 0.1*1.259*1.2*0.861*1.0013*1.02*0.991*1.032*0.976*0.945*1.057,
        "pi_len": 40,
        "90_corr": 1.0255*1.02*0.98
    },
    "q2":{
        "q_freq": 4.8933-0.569e-3,#4.8885+0.014e-3+0.95e-3+0.013e-3-0.265e-3+0.073e-3-0.05e-3+0.224e-3+0.06e-3,#5.0354-0.409e-3,#5.069-0.242e-3,#5.073-2.675e-3-0.031e-3, # GHz
        "LO": 5., # GHz
        "pi_amp": 0.1*0.862*0.9681*1.071*1.058*1.332*1.037*1.098*0.881*0.823*1.064,
        "pi_len": 40,
        "90_corr": 1.0
    },
    "q3":{   
        "q_freq": 4.7009+1.687e-3-0.016e-3-0.154e-3+0.149e-3-0.008e-3+0.028e-3,#second:4.5967 # GHz
        "LO": 4.850, # GHz
        "pi_amp": 0.1*0.798*1.035*0.936*1.02*1.355*0.927*1.025*1.041*1.0195*0.9888*1.03,
        "pi_len": 40,
        "90_corr": 1.0207*0.9723*1.06*0.9656
    },
    "q4":{
        "q_freq": 5.078-2.178e-3-0.137e-3+0.002e-3+0.011e-3,#second:4.9729 # GHz
        "LO": 5.25, # GHz
        "pi_amp": 0.1*1.572*1.0526*0.788*1.14*1.03*0.9389*1.2*1.034,#*1.221,
        "pi_len": 40,
        "90_corr": 1.0734*0.9523*0.9754*1.0483*0.9798
    },
    "q5":{
        "q_freq": 7.57, # GHz
        "LO": 7.62, # GHz
        "pi_amp": 0.1,
        "pi_len": 400,
        "90_corr": 1.0
    },
    "q6":{
        "q_freq": 7.323, # GHz
        "LO": 7.36, # GHz
        "pi_amp": 0.1,
        "pi_len": 400,
        "90_corr": 1.0
    },
    "q7":{
        "q_freq": 8.0377, # GHz
        "LO": 8.1, # GHz
        "pi_amp": 0.1,
        "pi_len": 400,
        "90_corr": 1.0
    },
    "q8":{
        "q_freq": 7.9224, # GHz
        "LO": 7.965, # GHz
        "pi_amp": 0.2*1.2*0.95,
        "pi_len": 240,
        "90_corr": 1.0
    },
}    
updating_qubit = ["q0","q1","q2","q3","q4","q5","q6","q7","q8"]

# name, q_freq(GHz), LO(GHz), amp, len, half
# update_info = [['q1', 4.526-0.005, 4.60, 0.2*0.9*1.05*1.01, 40, 1.03 ]]
for i in updating_qubit:
    # wiring = spec.get_spec_forConfig('wire')
    q_name = i
    qubit_LO = xy_infos[i]["LO"]
    qubit_RF = xy_infos[i]["q_freq"]
    ref_IF = (qubit_RF-qubit_LO)*1000

    print(f"center {ref_IF} MHz")
    pi_amp = xy_infos[i]["pi_amp"]
    pi_len = xy_infos[i]["pi_len"]

    qubit_wf = xy_infos[i]["wf"] if "wf" in xy_infos[i].keys() else 0

    update_controlFreq(config_obj, spec.update_aXyInfo_for(target_q=q_name,IF=ref_IF,LO=qubit_LO))
    if np.abs(ref_IF) > 350:
        print("Warning IF > +/-350 MHz, IF is set 350 MHz")
        ref_IF = np.sign(ref_IF)*350

    spec.update_aXyInfo_for(target_q=q_name, 
                            amp=pi_amp, 
                            len=pi_len, 
                            half=xy_infos[i]["90_corr"], 
                            wf=qubit_wf
                            )
    update_controlWaveform(config_obj, spec.get_spec_forConfig("xy"), target_q=q_name )

from QM_driver_AS.ultitly.config_io import output_config
output_config( link_path, config_obj, spec )
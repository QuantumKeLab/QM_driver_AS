# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer
init_macro = initializer(200000,mode='wait')

from exp.save_data import save_nc, save_fig, create_folder

import matplotlib.pyplot as plt
import numpy as np
# Set parameters
init_macro = initializer(300000,mode='wait')

ro_element = ["q3_ro","q4_ro"]
flux_Qi = 4
excited_Qi = [3,4]

n_avg = 200

time_max = 0.1 # us
time_resolution = 0.004 # us
amps_max = 0.3
amps_resolution = 0.003
save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"q{flux_Qi}_czchavron"
folder_label = "cz_chavron_1"
from exp.cz_chavron import CZ
dataset = CZ(time_max,time_resolution,amps_max,amps_resolution,ro_element,flux_Qi,excited_Qi,qmm,config,n_avg=n_avg,initializer=init_macro)

if save_data: 
    save_dir = create_folder(save_dir, folder_label)
    save_nc(save_dir, save_name, dataset) 

# Plot
from exp.plotting import plot_and_save_cz_chavron
plot_and_save_cz_chavron(dataset, save_dir, save_data )
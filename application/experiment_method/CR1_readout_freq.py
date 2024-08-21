

# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer

from exp.save_data import save_nc, save_fig, create_folder

import matplotlib.pyplot as plt

from exp.readout_optimization import ROFreq
my_exp = ROFreq(config, qmm)
my_exp.initializer = initializer(300000,mode='wait')
my_exp.ro_elements = ["q3_ro", "q4_ro"]
my_exp.xy_elements = ['q4_xy']
my_exp.freq_range = (-5, 5)
my_exp.freq_resolution = 0.1
my_exp.preprocess = "shot"
save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"ro_amp_{my_exp.xy_elements[0]}"
folder_label = "readout_freq_1"
# Start measurement
dataset = my_exp.run(1000)

# Data Saving 
if save_data: 
    folder_save_dir = create_folder(save_dir, folder_label)
    save_nc( folder_save_dir, save_name, dataset)

# Plot
from exp.readout_optimization import *
if my_exp.preprocess == "shot":
    dataset = dataset.transpose("mixer","shot","prepare_state","frequency")

else:
    dataset = dataset.transpose("mixer","prepare_state","frequency")
from exp.plotting import plot_and_save_readout_freq
plot_and_save_readout_freq(dataset, my_exp, folder_save_dir, save_data)
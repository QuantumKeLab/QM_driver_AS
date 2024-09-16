# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer

import matplotlib.pyplot as plt

# Set parameters
init_macro = initializer(200000,mode='wait')

ro_elements = ["q3_ro", "q4_ro"]
q_name =  "q3_xy"


save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"{q_name}_XYampCali"
folder_label = "xy_amp_1"
n_avg = 500

sequence_repeat = 1
amp_modify_range = 0.4/float(sequence_repeat)
from exp.SQGate_calibration import *
drag_coef = 0.5
# output_data = DRAG_calibration_Yale( drag_coef, q_name, ro_element, config, qmm, n_avg=n_avg)
dataset = amp_calibration( amp_modify_range, q_name, ro_elements, config, qmm, n_avg=n_avg, sequence_repeat=sequence_repeat, simulate=False, mode='live')


#Save data
save_data = 1
folder_label = "xy_amp" #your data and plots will be saved under a new folder with this name
if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,folder_label)

# Plot
from exp.plotting import plot_and_save_xy_amp
plot_and_save_xy_amp(dataset, folder_label, save_data)

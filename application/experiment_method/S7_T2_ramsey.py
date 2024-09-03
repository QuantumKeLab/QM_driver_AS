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
import numpy as np
import matplotlib.pyplot as plt
from exp.ramsey import Ramsey

import numpy as np

#Set parameters
my_exp = Ramsey(config, qmm)
from ab.QM_config_dynamic import initializer
my_exp.initializer = initializer(50000,mode='wait')
my_exp.ro_elements = ["q3_ro"]
my_exp.xy_elements = ["q3_xy"]
my_exp.virtual_detune = 1
my_exp.max_time = 8
my_exp.time_resolution = 0.02
dataset = my_exp.run(400)


from exp.plotting import plot_and_save_t2_ramsey_singleRun, plot_and_save_t2_repeateRun

#Save data
save_data = 1
if save_data: 
    from exp.save_data import DataPackager
    folder_label = "ramseyT2_stat" #your data and plots will be saved under a new folder with this name
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,"ramseyT2_stat")

# Plot

# time = (dataset.coords["time"].values)/1000
# plot_and_save_t2_ramsey_singleRun(dataset, time, folder_save_dir, save_data)

#Repetition T2
from exp.repetition_measurement import RepetitionMeasurement
re_exp = RepetitionMeasurement()
re_exp.exp_list = [my_exp]
re_exp.exp_name = ["T2"]
my_exp.shot_num = 400
dataset = re_exp.run(10)

save_data = 1
if save_data: 
    from exp.save_data import DataPackager
    folder_label = "ramseyT2_rep" #your data and plots will be saved under a new folder with this name
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,"ramseyT2_rep")

#To plot the result of multiple measurements (2D graph and histogram), use the following block of code
#================================================================================================#
import qcat.visualization.qubit_relaxation as qv
print(dir(qv))
dataset.data_vars.items()
single_name = "q1_ro"
plot_and_save_t2_ramsey_repeateRun(dataset, time, single_name, folder_save_dir, save_data)

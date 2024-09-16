
# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer

from exp.save_data import save_nc, save_fig

import matplotlib.pyplot as plt

# Set parameters
init_macro = initializer(200000,mode='wait')

ro_elements = ["q0_ro", "q1_ro", "q2_ro", "q3_ro", "q4_ro"]
q_name =  ["q4_xy"]
save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"{q_name[0]}_XYfreqCali"

n_avg = 200  # Number of averages
virtual_detune = 5 # Unit in MHz

# Start measurement
from exp.ramsey import Ramsey
my_exp = Ramsey(config, qmm)
my_exp.ro_elements = ["q3_ro", "q4_ro"]
my_exp.xy_elements = ['q3_xy']
my_exp.initializer=initializer(10000,mode='wait')

dataset = my_exp.run( 20 )
# dataset = ramsey_freq_calibration( virtual_detune, q_name, ro_elements, config, qmm, n_avg=n_avg, simulate=False, initializer=init_macro)


if save_data: save_nc( save_dir, save_name, dataset)

# for ro_element, data in output_data.items():
#     plot_ana_result(evo_time,data[0],virtual_detune)
plot_data = dataset["q4_ro"].values[0]
evo_time = dataset.coords["time"].values
plot_ana_result(evo_time,plot_data,virtual_detune)
    
if save_data: save_fig(save_dir, save_name)
plt.show()


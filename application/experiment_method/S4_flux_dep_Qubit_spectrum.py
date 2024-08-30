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
from exp.plotting import plot_and_save_flux_dep_Qubit

# Start meausrement
from exp.xyfreq_sweep_flux_dep import XYFreqFlux
my_exp = XYFreqFlux(config, qmm)
my_exp.ro_elements = ["q3_ro"]
my_exp.xy_elements = ['q3_xy']
my_exp.z_elements = ['q3_z']
my_exp.initializer=initializer(10000,mode='wait')
my_exp.xy_driving_time = 20
my_exp.xy_amp_mod = 0.02
my_exp.z_amp_ratio_range = (-0.02,0.02)
my_exp.z_amp_ratio_resolution = 0.02
my_exp.freq_range = (-400,400)
my_exp.freq_resolution = 4
my_exp.sweep_type = "overlap"
dataset = my_exp.run( 500 )

save_data = 1
if save_data: 
    from exp.save_data import DataPackager
    folder_label = "Flux_dep_Qubit" #your data and plots will be saved under a new folder with this name
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,"Flux_dep_Qubit")

# Plot
plot_and_save_flux_dep_Qubit(dataset, save_dir, save_data)
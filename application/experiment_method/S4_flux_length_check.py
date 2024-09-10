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
from exp.flux_length_check import Flux_length_check
my_exp = Flux_length_check(config, qmm)
my_exp.ro_elements = ["q3_ro","q4_ro"]
my_exp.xy_elements = ['q4_xy']
my_exp.z_elements = ['q4_z']

my_exp.initializer=initializer(10000,mode='wait')

my_exp.flux_type = "offset" #offset or play

my_exp.xy_driving_time = 10
my_exp.xy_amp_mod = 0.1

my_exp.flux_quanta = 0.2
my_exp.set_flux_quanta = 0.1

my_exp.freq_range = (-150,50)
my_exp.freq_resolution = 1 

dataset = my_exp.run( 100 )

#Save data
save_data = 1
folder_label = "Flux_check" #your data and plots will be saved under a new folder with this name
if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,folder_label)

# Plot
save_figure = 1
from exp.plotting import PainterFluxCheck
painter = PainterFluxCheck()
figs = painter.plot(dataset,folder_label)
if save_figure: dp.save_figs( figs )
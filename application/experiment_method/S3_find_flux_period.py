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
from exp.plotting import plot_and_save_flux_period

# Set parameters
init_macro = initializer(10000,mode='wait')
ro_elements = ["q5_ro"]
z_elements = ['q5_z']

n_avg = 100
freq_range = (-5,5)
freq_resolution = 0.1 #0.1
flux_range = (-0.3,0.3)
flux_resolution = 0.02 #0.01

from exp.rofreq_sweep_flux_dep import *
dataset = freq_sweep_flux_dep(ro_elements, z_elements, config, qmm, freq_range=freq_range, freq_resolution=freq_resolution, flux_settle_time=1, flux_range=flux_range, flux_resolution=flux_resolution, n_avg=n_avg, initializer=init_macro)
# dataset = freq_sweep_flux_dep_stable(ro_elements, z_elements, config, qmm, freq_range=freq_range, freq_resolution=freq_resolution, flux_settle_time=1, flux_range=flux_range, flux_resolution=flux_resolution, n_avg=n_avg, initializer=init_macro)

#Save data
save_data = 1
folder_label = "Find_Flux_Period" #your data and plots will be saved under a new folder with this name
if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,folder_label)
    

# Plot
save_figure = 1
from exp.plotting import PainterFindFluxPeriod
painter = PainterFindFluxPeriod()
figs = painter.plot(dataset,folder_label)
if save_figure: dp.save_figs( figs )

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

from exp.save_data import save_nc, save_fig

from visualization.find_ZZfree_plot import plot_ZZfree, plot_tau_X_flux

import xarray as xr

# # Set parameters
# from exp.find_zzfree import find_ZZfree
# my_exp = find_ZZfree(config, qmm)
# my_exp.ro_elements = ["q2_ro"]
# my_exp.target_xy = ["q2_xy"]
# my_exp.crosstalk_xy = ["q3_xy"] # conditional x gate
# my_exp.coupler_z = ["q7_z"]
# my_exp.virtual_detune = 5
# my_exp.flux_range = ( -0.25, -0.05 )
# my_exp.resolution = 0.001


# my_exp.initializer = initializer(200000,mode='wait')
# dataset = my_exp.run( 1000 )

# save_data = True
# file_name = f"find_ZZfree_{my_exp.target_xy[0][:2]}_{my_exp.crosstalk_xy[0][:2]}"
# save_dir = link_config["path"]["output_root"]

# if save_data: save_nc( save_dir, file_name, dataset)

# Plot
dataset = xr.open_dataset(r"C:\Users\quant\SynologyDrive\09 Data\Fridge Data\Qubit\20240521_DR4_5Q4C_0430#7\20240620_DR4_5Q4C_0430#7\00 raw data\20240702_1712_find_ZZfree_q2_q3.nc")
plot_tau_X_flux(dataset) 
# if save_data: save_fig( save_dir, file_name)
plt.show()
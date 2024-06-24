# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer
init_macro = initializer(200000,mode='wait')

from exp.save_data import save_nc, save_fig

import matplotlib.pyplot as plt

from exp.zz_ramsey import plot_phase, plot_difference, plot_ana_result
# Set parameters
ro_elements = ["q3_ro"]
q_name = ['q3_xy']
con_xy_element = ["q4_xy"] # conditional x gate
coupler_element = ["q8_z"]

n_avg = 1000
virtual_detune = 5
flux_range = 0.1

save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"{q_name[0]}_zz ramsey"

from exp.zz_ramsey import exp_zz_ramsey
dataset = exp_zz_ramsey(20,0.04,flux_range,ro_elements,q_name,con_xy_element,coupler_element,n_avg,config,qmm,virtual_detune=virtual_detune,initializer=init_macro)

import xarray as xr

if save_data: dir = save_nc(save_dir, save_name, dataset)
dataset = xr.open_dataset(dir)

plot_data = dataset[ro_elements[0]].values[0]
time = dataset.coords["time"].values
flux = dataset.coords["flux"].values
frequency = dataset.coords["frequency"].values
fig, ax = plt.subplots()
# plot_phase(time, dataset1, dataset2,ax)
# plot_difference(time, dataset1, dataset2,ax)
# plot_detune(time, dataset)
plot_ana_result(time, flux, frequency, plot_data)

plt.legend()
plt.show()
if save_data: save_fig(link_config["path"]["output_root"], f"{q_name[0]}_zz ramsey")
# plot_difference(time, dataset1, dataset2,ax)
# plt.legend()
# plt.show()

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
init_macro = initializer(10000,mode='wait')

ro_elements = ["q0_ro", "q1_ro", "q2_ro", "q3_ro", "q4_ro"]
q_name = ['q0_xy']
z_name = ['q0_z']

sweep_type = "two_tone"      # "z_pulse", "const_z", "two_tone"

saturation_len = 20  # In us (should be < FFT of df)
saturation_ampRatio = 0.05 # pre-factor to the value defined in the config - restricted to [-2; 2)
n_avg = 1000

flux_range = (0.0,0.15)
flux_resolution = 0.01

freq_range = (-5,5)
freq_resolution = 0.1


# Start meausrement
from exp.xyfreq_sweep_flux_dep import XYFreqFlux
my_exp = XYFreqFlux(config, qmm)
my_exp.ro_elements = ["q8_ro", "q3_ro", "q4_ro"]
my_exp.xy_elements = ['q8_xy']
my_exp.z_elements = ['q8_z']
my_exp.initializer=initializer(10000,mode='wait')
my_exp.sweep_type = "overlap"
my_exp.xy_driving_time = 20
my_exp.z_amp_ratio_range = (-0.2,0.2)
my_exp.z_amp_ratio_resolution = 0.01
my_exp.freq_range = (-150,50)
my_exp.freq_resolution = 1.0
dataset = my_exp.run( 500 )

save_data = False
save_name = f"Spectrum_{q_name[0]}_{z_name[0]}_{sweep_type}"
save_dir = link_config["path"]["output_root"]
if save_data: save_nc( save_dir, save_name, dataset)

# Plot
from exp.xyfreq_sweep_flux_dep import plot_ana_flux_dep_qubit
freqs = dataset.coords["frequency"].values
flux = dataset.coords["amp_ratio"].values
for i, (ro_name, data) in enumerate(dataset.data_vars.items()):
    xy_LO = dataset.attrs["xy_LO"][0]/1e6
    xy_IF_idle = dataset.attrs["xy_IF"][0]/1e6
    z_offset = dataset.attrs["z_offset"][0]
    print(ro_name, xy_LO, xy_IF_idle, z_offset, data.shape)
    fig, ax = plt.subplots(2)

    plot_ana_flux_dep_qubit(data, flux, freqs, xy_LO, xy_IF_idle, z_offset, ax)
    # plot_ana_flux_dep_qubit_1D(data, flux, freqs, xy_LO, xy_IF_idle, z_offset, ax) 


    ax[0].set_title(ro_name)
    ax[1].set_title(ro_name)

if save_data: save_fig(save_dir, save_name)

plt.show()
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

# Set parameters
from exp.readout_optimization import ROFreqAmpMapping
my_exp = ROFreqAmpMapping(config, qmm)
my_exp.initializer = initializer(300000,mode='wait')

my_exp.ro_elements = [ "q3_ro", "q4_ro"]
my_exp.xy_elements = ['q4_xy']

my_exp.freq_range = (-3, 3)
my_exp.freq_resolution = 0.1

my_exp.amp_mod_range = (0.5, 1.5)
my_exp.amp_resolution = 0.05

dataset = my_exp.run( 1000 )

save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"ro_map_{my_exp.xy_elements[0]}"
folder_label = "ro_mapping_1"

if save_data: 
    folder_save_dir = create_folder(save_dir, folder_label)
    save_nc(save_dir, save_name, dataset)


# Plot
from exp.plotting import plot_and_save_readout_mapping
plot_and_save_readout_mapping(dataset, folder_save_dir, save_data)




 
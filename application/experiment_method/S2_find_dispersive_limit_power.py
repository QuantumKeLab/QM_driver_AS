# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer

from exp.plotting import plot_and_save_dispersive_limit

# Set parameters
from exp.rofreq_sweep_power_dep import ROFreqSweepPowerDep
my_exp = ROFreqSweepPowerDep(config, qmm)
my_exp.initializer = initializer(10000,mode='wait')
my_exp.ro_elements = ["q3_ro","q4_ro"] #"q2_ro","q3_ro","q4_ro","q5_ro",
my_exp.freq_range = (-10,10)
my_exp.freq_resolution = 0.05
my_exp.amp_mod_range = (0,2) # tha value range >0, <2
my_exp.amp_scale = "lin"
dataset = my_exp.run( 20 )

#Save data
save_data = 1
folder_label = "power_dep_resonator" #your data and plots will be saved under a new folder with this name
if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,folder_label)

# Plot
save_figure = 1
from exp.plotting import PainterPowerDepRes
painter = PainterPowerDepRes()
figs = painter.plot(dataset,folder_label)
if save_figure: dp.save_figs( figs )

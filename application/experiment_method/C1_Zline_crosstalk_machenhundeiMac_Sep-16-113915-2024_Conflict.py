# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer

# from exp.save_data import save_fig, create_folder
save_dir = link_config["path"]["output_root"]

import matplotlib.pyplot as plt

from visualization.zline_crosstalk_plot import plot_crosstalk_3Dscalar, plot_analysis



import xarray as xr

# Set parameters
from exp.zline_crosstalk import FluxCrosstalk
my_exp = FluxCrosstalk(config, qmm)
my_exp.detector_qubit = "q3"
my_exp.detector_is_coupler = False
my_exp.crosstalk_qubit = "q4"
my_exp.ro_elements = ["q3_ro", "q4_ro"]

my_exp.expect_crosstalk = 0.1
my_exp.detector_bias = 0.
my_exp.z_modify_range = 0.2
my_exp.z_resolution = 0.008
my_exp.z_time = 0.16

my_exp.measure_method = "long_drive"   #long_drive, ramsey
my_exp.z_method = "pulse"     #offset, pulse

my_exp.initializer = initializer(50000,mode='wait')
dataset = my_exp.run( 100 )


save_data = True
folder_label = "Zline_crosstalk_1" #your data and plots with be saved under a new folder with this name
file_name = f"detector_{my_exp.detector_qubit}_crosstalk_{my_exp.crosstalk_qubit}_{my_exp.measure_method}_{my_exp.z_method}_expectcrosstalk_{my_exp.expect_crosstalk}_{my_exp.z_time}mius"
if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,"Zline_crosstalk_1")

# Plot
analysis_figures = plot_analysis(dataset)
raw_figures = plot_crosstalk_3Dscalar(dataset)
from exp.save_data import DataPackager

dp.save_figs(raw_figures)
# dp.save_figs(analysis_figures)

# # 保存每个图像并显示
# for fig, ax, q in raw_figures:
#     plt.figure(fig.number)  # 设置当前图形对象
#     if save_data:
#         save_fig( save_dir, f"{q}_{file_name}_raw")

# for fig, ax, q in analysis_figures:
#     plt.figure(fig.number)  # 设置当前图形对象
#     if save_data:
#         save_fig( save_dir, f"{q}_{file_name}_analysis")


# for fig, ax, q in raw_figures:
#     plt.figure(fig.number)  # 设置当前图形对象
#     plt.show()  # 显示图像
# for fig, ax, q in analysis_figures:
#     plt.figure(fig.number)  # 设置当前图形对象
#     plt.show()  # 显示图像


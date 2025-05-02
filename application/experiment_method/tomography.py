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
from qm.qua import *
import numpy as np

# Set parameters
ro_element = ["q2_ro","q3_ro","q4_ro"]
q_name = ["q2_xy","q3_xy","q4_xy"]
n_avg = 1000


def prepare_state():
    
    
    play("x180","q2_xy")
    align()
    play("y90", "q2_xy")
    align()
    play("x180","q3_xy")
    align()
    play("y90", "q3_xy" )
    align()
    play("const" * amp(-0.046),"q3_z", duration=25)
    play("const"*amp(0.30),"q7_z",duration=25)
    align()
    frame_rotation_2pi(-3.0718456925739788/(2*np.pi), "q3_xy")
    frame_rotation_2pi(1.7003101134811742/(2*np.pi), "q2_xy")
    align()
    play("x180","q3_xy")
    align()
    play("y90", "q3_xy" )
    #play("x180","q2_xy")
    #play("x180","q3_xy")

    pass

from exp.tomography import StateTomography, calculate_2Q_block_vector, calculate_block_vector, calculate_density_matrix, plot_density_matrix
my_exp = StateTomography(config, qmm)
my_exp.ro_elements = ["q0_ro","q1_ro","q2_ro","q3_ro","q4_ro"]
my_exp.xy_elements = ["q3_xy","q2_xy"]
threshold1 = 3.275e-5#1.647e-4#6.840e-5#4.593e-5
threshold2 = 1.619e-4#1.434e-5#1.376e-4
my_exp.process = prepare_state
dataset = my_exp.run(1000)

save_data = True
save_dir = link_config["path"]["output_root"]
save_name = f"{ro_element[1]}{ro_element[2]}_state_tomography"


#dataset = state_tomography_NQ(q_name,ro_element,prepare_state,n_avg,config,qmm,simulate=False)
folder_label = "Tomography" #your data and plots will be saved under a new folder with this name

if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)
    dp.save_nc(dataset,folder_label) 

data_i1 = dataset[my_exp.ro_elements[2]].values[0].transpose((1,2,0))
data_i2 = dataset[my_exp.ro_elements[3]].values[0].transpose((1,2,0))
print(data_i1.shape)
vect_dis = calculate_2Q_block_vector(data_i1, threshold1, data_i2, threshold2)
density_matrix = calculate_density_matrix(data_i1, threshold1, data_i2, threshold2)
print(density_matrix[0,0],density_matrix[1,1],density_matrix[2,2], density_matrix[3,3])
fig,ax = plt.subplots()
plot_density_matrix(density_matrix)
plt.show()

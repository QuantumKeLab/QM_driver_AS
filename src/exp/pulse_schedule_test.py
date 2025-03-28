# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent.parent/"application"/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()
# print(config["waveforms"]["q0_xy_xcos180_wf_I"])
from ab.QM_config_dynamic import initializer

from exp.cryoscope import Cryoscope
from exp.exp_temp import ExpTemp
from exp.zline_crosstalk import FluxCrosstalk

from exp.randomized_banchmarking_sq_z import randomized_banchmarking_sq
my_exp = randomized_banchmarking_sq(config, qmm)
my_exp.initializer = initializer(2000,mode='wait')
my_exp.z_elements = ["q6_z"]

# my_exp.gate_length = 16
# my_exp.sweep_type = "overlap"
my_exp.pulse_schedule_simulation(["con1"],20000)
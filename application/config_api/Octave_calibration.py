"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
You need to run this file in order to update the Octaves with the new parameters.
"""
from QM_driver_AS.ultitly.set_octave import ElementsSettings, octave_settings

# Configure the Octave parameters for each element
rr0 = ElementsSettings("q0_ro", gain=2, rf_in_port=["octave1", 1], down_convert_LO_source="Internal")


q0_xy = ElementsSettings("q0_xy", gain=18)
q1_xy = ElementsSettings("q1_xy", gain=18)
q2_xy = ElementsSettings("q2_xy", gain=18)
q3_xy = ElementsSettings("q3_xy", gain=18)
q4_xy = ElementsSettings("q4_xy", gain=18)
q5_xy = ElementsSettings("q5_xy", gain=18)
# q6_xy = ElementsSettings("q6_xy", gain=18)
# q7_xy = ElementsSettings("q7_xy", gain=18)
q8_xy = ElementsSettings("q8_xy", gain=18)

# q2_xy = ElementsSettings("q2_xy", gain=15)

# Add the "octave" elements
# elements_settings = [rr0]
elements_settings = [q3_xy,]


###################
# Octave settings #
###################
# Configure the Octave according to the elements settings and calibrate
 
# Dynamic config
from pathlib import Path
# Get the current file path
current_file = Path(__file__).resolve()
# Get the parent directory
link_path = current_file.parent/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config
config_obj, spec = import_config( link_path )
qmm, octaves = spec.buildup_qmm()

config = config_obj.get_config()
# dyna_config.check_mixerCorrectionPair_for('q1')

octave_settings(
    qmm=qmm,
    config=config,
    octaves=octaves,
    elements_settings=elements_settings,
    calibration=True,
)
qmm.close()


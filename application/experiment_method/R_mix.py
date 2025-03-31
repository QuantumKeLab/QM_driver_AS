# Import necessary file
from pathlib import Path
link_path = Path(__file__).resolve().parent.parent/"config_api"/"config_link.toml"

from QM_driver_AS.ultitly.config_io import import_config, import_link
link_config = import_link(link_path)
config_obj, spec = import_config( link_path )

config = config_obj.get_config()
qmm, _ = spec.buildup_qmm()

from ab.QM_config_dynamic import initializer

from exp.readout_fidelity import ROFidelity
from qcat.visualization.readout_fidelity import plot_readout_fidelity
from qcat.analysis.state_discrimination.readout_fidelity import GMMROFidelity
from exp.relaxation_time import exp_relaxation_time
from qcat.analysis.qubit.relaxation import  RelaxationAnalysis
from exp.single_spin_echo import SpinEcho
from qcat.analysis.qubit.relaxation import  RelaxationAnalysis



single_shot = ROFidelity(config, qmm)
single_shot.initializer = initializer(1500000,mode='wait')
single_shot.ro_elements = ["q0_ro", "q1_ro", "q2_ro", "q3_ro"]
single_shot.xy_elements = ['q0_xy']

#Set parameters
T1 = exp_relaxation_time(config, qmm)
T1.initializer = initializer(1000000,mode='wait')
T1.ro_elements = ["q0_ro", "q1_ro", "q2_ro", "q3_ro"]
T1.xy_elements = ["q0_xy"]
T1.max_time = 1000
T1.time_resolution = 10

# Set parameters
T2 = SpinEcho( config, qmm )
T2.initializer = initializer(1000000,mode='wait')
T2.ro_elements = ["q0_ro", "q1_ro", "q2_ro", "q3_ro"]
T2.xy_elements = ["q0_xy"]
T2.time_range = ( 40, 500000 )
T2.time_resolution = 5000

save_data = True
save_dir = link_config["path"]["output_root"]
folder_label = "mix"
# Start measurement

# Data Saving 
if save_data: 
    from exp.save_data import DataPackager
    save_dir = link_config["path"]["output_root"]
    dp = DataPackager( save_dir, folder_label )
    dp.save_config(config)


    for i in range(200):
        dataarray_1 = single_shot.run(10000)
        dp.save_nc(dataarray_1,f"{i}_shot")
        dataarray_2 = T1.run(400)
        dp.save_nc(dataarray_2,f"{i}_T1")
        dataarray_3 = T2.run(1000)
        dp.save_nc(dataarray_2,f"{i}_T2")




        for ro_name in dataarray_1.coords["q_idx"].values:
            datas = dataarray_1.sel(q_idx=ro_name).drop_vars("q_idx")
            my_ana = GMMROFidelity()
            my_ana._import_data(datas)
            # print(my_ana.raw_data)
            try:
                my_ana._start_analysis()

                fig = plot_readout_fidelity(datas, my_ana, my_ana.export_G1DROFidelity(), plot=False)

                dp.save_fig( fig, f"{i}_shot_{ro_name}" )
            except:
                print(f"{ro_name} fail")


        figs = []
        for ro_name in dataarray_2.coords["q_idx"].values:
            data = dataarray_2.sel(q_idx=ro_name)
            data.attrs = dataarray_2.attrs
            data.name = ro_name
            my_ana = RelaxationAnalysis(data.sel(mixer="I"))
            my_ana._start_analysis()
            figs.append((f"{i}_T1_{data.name}",my_ana.fig))

        dp.save_figs( figs )

        figs = []
        for ro_name in dataarray_3.coords["q_idx"].values:
            data = dataarray_3.sel(q_idx=ro_name)
            data.attrs = dataarray_3.attrs
            data.name = ro_name
            my_ana = RelaxationAnalysis(data.sel(mixer="I"))
            my_ana._start_analysis()
            figs.append((f"{i}_T2_{data.name}",my_ana.fig))

        dp.save_figs( figs )
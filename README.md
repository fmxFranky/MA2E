Running Script
* bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)
* Example : bash run.sh qmix sc2 3s_vs_5z use_MT=True 3 0 3 => Run qmix+MA2E in SMAC 3s_vs_5z scenario

'use_MT' arg means executing the model taht plugs in MA2E into the baseline algorithm. 
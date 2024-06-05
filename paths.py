import os

name = os.popen('whoami').read()

if name == 'laptop-5i5qsv76\\joaco\n':  # Asus Rog
    main_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEDYN/'

elif name == 'desktop-r5hd7ct\\joac\n':  # Liaa Colores
    main_path = 'D:/OneDrive - The University of Nottingham/MEGEYEDYN/'

elif name == 'desktop-59r7a1d\\usuario\n':  # Desktop
    main_path = 'C:/Users/Usuario/OneDrive - The University of Nottingham/MEGEYEDYN/'

elif name == 'ad\\lpajg1\n':  # Notts
    main_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEDYN/'

else:
    scripts_path = os.path.dirname(os.path.realpath(__name__))
    main_path = scripts_path.replace('Scripts', '')

    # Check that DATA folder exists in main_path
    data_path_exists = os.path.exists(main_path + 'DATA/')
    if not data_path_exists:
        raise AssertionError(f'DATA/ folder not found in main path: {main_path}\n'
                             f'Please refer to https://github.com/jegonza66/MEGEYEDYN ')

ctf_path = main_path + 'DATA/CTF_DATA/'
et_path = main_path + 'DATA/ET_DATA/'
bh_path = main_path + 'DATA/BH_DATA/'
mri_path = main_path + 'DATA/MRI_DATA/'
opt_path = main_path + 'DATA/OPT_DATA/'
exp_path = main_path + 'Experiment/'


save_path = main_path + 'Save/'
os.makedirs(save_path, exist_ok=True)

preproc_path = main_path + 'Save/Preprocessed_Data/'
os.makedirs(preproc_path, exist_ok=True)

filtered_path = main_path + 'Save/Filtered_Data_RAW/'
os.makedirs(filtered_path, exist_ok=True)

filtered_path = main_path + 'Save/Filtered_Data_ICA/'
os.makedirs(filtered_path, exist_ok=True)

ica_path = main_path + 'Save/ICA_Data/'
os.makedirs(ica_path, exist_ok=True)

# results_path = main_path + 'Results/'
# os.makedirs(results_path, exist_ok=True)

plots_path = main_path + 'Plots/'
os.makedirs(plots_path, exist_ok=True)

sources_path = main_path + 'Save/Source_Data/'
os.makedirs(sources_path, exist_ok=True)

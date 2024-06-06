# MEGEYEDYN Project

## A MEG / Eye Tracker Coregistration Study

### Project Structure

To run the scripts from this project please set the directory as follows:

```bash
├───DATA
│   ├───CTF_DATA
│   │   ├───13229005
│   │   │
│   │   ├───13703055
│   │   │   
│   │   ├───15584005
│   │   │               
│   │   └───BACK_NOISE
│   │       ├───15644002_MatiasIson_20240327_01.ds
│   │       │       
│   │       ├───15644002_MatiasIson_20240327_02.ds
│   │       │       
│   │       └───17439002_MatiasIson_noise_20240408_01.ds
│   │               
│   ├───ET_DATA
│   │   ├───13229005
│   │   │           
│   │   ├───13703055
│   │   │           
│   │   └───15584005
│   │              
│   └───OPT_DATA
│       ├───13229005
│       │       
│       ├───13703055
│       │       
│       └───15584005
│               
├───Experiment
│   ├───Data Viewer Results
│   │                   
│   ├───Images
│   │       
│   ├───Videos
│   │           
│   └───WebLink                 
│       
└───Scripts
    └───preprocessing.py
```

Brief explanation of the main modules:

- paths.py: Module including paths to data and save directories.
- setup.py: Module defining:
    - Experiment information in the exp_info class, that contains information about every subjects scanning.
    - Subject object in raw_subject class, that will be used in the preprocessing module and will store information about the subjects preprocessing, scanning and
      behabioural and Eye-tracker data.
- Preprocessing.py (In progress) module will run the preprocessing of all subjects listed in the exp_info.subjects_ids from setup.py using functions from
  funcitons_preproc.py.
- functions_preproc.py: Module containing functions for preprocessing the dataset, with functions including scaling ET channels in MEG signal, running eye-movements
  detection with Remodnav algorithm, etc.
- plot_preproc.py: Module with ploting functions of preprocessing results.
- artifact_annot.py: Module to run a manual artifact annotation on the MEG signal to reject noisy intervals and channels
- clean_ica.py: Module to run ICA artifact remouval.
- epoch_evoked_ga.py: Module to run epoching and average to evoked response of every participant, and computing Grand Average of evoked data. Options to save and plot
  epoched and evoked results.
- plot_general.py: Module with plotting functions for main analysis.
- save.py and load.py: Modules to save and load variables, figures, and objects (preprocessed subjects and MEG data).
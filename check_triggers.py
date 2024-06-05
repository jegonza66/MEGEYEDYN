import setup

exp_info = setup.exp_info()
config = setup.config()

subject_code = exp_info.subjects_ids[-1]

subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)

meg_data = subject.load_raw_meg_data()

meg_data.pick(['UPPT001', 'UPPT002'])

meg_data.plot()
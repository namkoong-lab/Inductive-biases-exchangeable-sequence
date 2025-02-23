# Autoregressive Sequence Model for Active Learning

This repository contains the code for the project "Thompson Sampling with Autoregressive Model for Contextual Bandit".
Due to diverging codebase, this is a duplicated version of the parent folder.

Here are the steps you need to take to reproduce the active learning results

1. Generate the data: Run `python ./data/generate_data.py --data_dir [where_to_save_data] --data_type al_regions_v2`
2. Train models on the generated data:

   - modify `dataset_dir` field in `./scripts/uq_al_regions_v2_50_autoreg.yaml` and `./scripts/uq_al_regions_v2_50_excg.yaml`
   - run `python launch.py train.py [either yaml file above] --port 29500`

3. Run active learning experiments
   - modify `save_folder` on line 155 of `./active_learn/run_al.py` to where you want to save active learning results
   - modify `DATA_FOLDER` on line 210 of `./active_learn/run_al.py` to where the generated data (from step 1) is
   - run `./active_learn/run_al.py`

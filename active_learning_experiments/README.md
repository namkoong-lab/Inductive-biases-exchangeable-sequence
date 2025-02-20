# Thompson Sampling with Autoregressive Model for Contextual Bandit

This repository contains the code for the project "Thompson Sampling with Autoregressive Model for Contextual Bandit". 

## File Structure

- `data`
    - `load_data.py`: Define your training function classes here, see the GP constant for an example.

- `inference`
    - `ccb_inf_batch.py`: The bandit interaction code.

- `models`
    - `autoreg_model.py`: Contains `Autoreg_Model` class, which is the implementation of autoregressive model.
    - `excg_model.py`: Contains `ExCg_Model` class, which is the implementation of exchangeable model.
    - `TS_model.py`: The implementation of Thompson Sampling model, which takes in a UQ Transformer model.
    - `modules.py`: Some utility functions for the models.
    - `LinUCB.py`: The implementation of LinUCB.

- `scripts`
    - `train.py`: Main training scripts. 
    - `accelerate_config.yaml`: The configuration file for using distributed training with accelerate.
    - `uq_normal_autoreg.yaml`: The configuration file for using autoregressive model.
    - `uq_normal_excg.yaml`: The configuration file for using exchangeable model.
    - `launch.py`: launch the training scripts with config files.
    - `run_cb_train.py`: Directly run training for both models.

- `trainers`
    - `trainer.py`: Contains the distributed training and evaluation code.

- `utils`
    - `load_model.py`: Loading the model.
    - `scheduler.py`: The learning rate scheduler.

## Running the code

To run the training scripts, you can run: 
```
python launch.py train.py config.yaml --port 29500
```

Notice that you must use a different port for each training process, if you are running multiple processes on the same
machine. 

## Configurations

### Model Arguments (`model_args`)

- **dim_llm_embedding**: Dimensionality of input X.
- **dim_y**: Dimensionality of the Y (default: 1).
- **emb_depth**: Depth of the embeder that embeds the input X into a embedding to input into the transformer model (default: 1).
- **d_model**: Dimensionality of the transformer model input 
- **dim_feedforward**: Dimensionality of the feedforward network inside the transformer model
- **nhead**: Number of attention heads (must be a divisor of d_model) 
- **dropout**: Dropout rate (default: 0.1).
- **activation**: Activation function used in the model (default: gelu).
- **num_layers**: Number of layers of transformer stacks
- **bound_std**: Determins if bounding std output or not. Default bounding value is 0.05 (default: true).
- **embed_type**: Type of embedding used, either embedding the concatenated input X and Y or embedding the input X only. (default: embed_concat).
- **uncertainty**: Type of uncertainty modeling, can be `normal` or `riemann` (default: normal).
- **loss_type**: Type of loss function used, can be `logprob` or `mse` (default: logprob).
- **pad_value**: Padding value used in sequences (default: 0.0).
- **gradient_type**: Type of parts for freezing. if `full`, all the transformer layers are frozen. If `std`, then freeze
  all parameters, and only tune the std prediction network. (default: full).
- **model_type**: Specifies the model type, can be `autoreg` or `excg` 

### Training Arguments (`training_args`)

- **lr**: Learning rate (default: 0.0003).
- **seed**: Random seed for reproducibility (default: 3004).
- **weight_decay**: Weight decay for regularization (default: 0.01).
- **warmup_ratio**: Ratio of warmup steps, used for learning rate scheduler (default: 0.03).
- **min_lr**: Minimum learning rate, used for learning rate scheduler (default: 0.00003).
- **total_train_batch_size**: Total batch size for training, must be a multiple of `num_process` (default: 64).
- **total_test_batch_size**: Total batch size for testing, must be a multiple of `num_process` (default: 64).
- **num_process**: Number of processes for distributed training (default: 8).
- **epochs**: Number of training epochs (default: 400).
- **eval_steps**: Steps between evaluations. Since we are using data generated on the fly, this parameter is not used. 
- **train_horizon**: Training horizon (default: 2000).
- **test_horizon**: Testing horizon (default: 200).
- **eval_func**: Evaluation function name (default: "eval_func").
- **num_train_samples**: Number of training samples. This effectively controls the number of data in a training epoch. (default: 8192).
- **num_test_samples**: Number of testing samples (default: 8192).
- **load_from_checkpoint**: Load from checkpoint or not (default: false).
- **checkpoint_path**: Path to the model checkpoint file.

### Data Arguments (`data_args`)

- **dataset_name**: Name of the dataset (default: "gp").
- **num_train_workers**: Number of workers for training data loading (default: 8).
- **num_test_workers**: Number of workers for testing data loading (default: 8).
- **noise_scale**: Scale of noise added to the data (default: 0.1).
- **dataset_dir**: Directory path to the dataset file. For GP, this is not used.
- **alpha**: Alpha parameter for data processing (default: 0.05).

### Logging Arguments (`logging_args`)

- **task**: Task name for logging 
- **wandb_project**: Project name for Weights & Biases logging
- **eval_log_step**: Steps between logging evaluations (default: 10000).


## Models

Most parts of `autoreg_model.py` and `excg_model.py` are the same. The main difference comes from how they create input
and masks for training. Take a look at `construct_causal_input`, `construct_causal_mask`, 
`construct_causal_prediction_mask` in `autoreg_model.py` and `construct_excg_input`, `construct_excg_mask`,
`construct_excg_prediction_mask` in `excg_model.py` for more details. 

## Inference

The inference code is in `ccb_inf_batch.py`. The code should be self-explanatory. 


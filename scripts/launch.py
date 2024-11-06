import sys
import yaml
import subprocess
import argparse

def get_args_str(hyperparameters):
    args_str = ''
    for arg, value in hyperparameters.items():
        args_str += f'--{arg}={value} '
    return args_str.strip()

def main(config_file, main_script, port):
    # Read the YAML configuration file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Extract the hyperparameters from the configuration
    model_args = config['model_args']
    training_args = config['training_args']
    data_args = config['data_args']
    logging_args = config['logging_args']

    # Combine the hyperparameters into a single dictionary
    hyperparameters = {**model_args, **training_args, **data_args, **logging_args}


    command = [
        "accelerate", "launch",
        "--config_file", "accelerate_config.yaml",
        "--main_process_port", str(port),
        main_script,
    ] + get_args_str(hyperparameters).split()
    subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch script with optional port specification.")
    parser.add_argument("main_script", help="Path to the main script to run")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("--port", type=int, default=29500, help="Port number (default: 29500)")

    args = parser.parse_args()

    main(args.config_file, args.main_script, args.port)
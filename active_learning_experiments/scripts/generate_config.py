import yaml

data_type = "al_regions_v2_50"

print(f"Generating configs for {data_type}")

template = "./uq_al-region_autoreg.yaml"

with open(template, "r") as f:
    config = yaml.safe_load(f)

config["data_args"]["dataset_name"] = data_type

# Save autoregressive config
config["model_args"]["model_type"] = "autoreg"
fname = f"./uq_{data_type}_autoreg.yaml"
print(f"Saving autoregressive config to {fname}")
with open(fname, "w") as f:
    yaml.dump(config, f)

# Save excg config
config["model_args"]["model_type"] = "excg"
fname = f"./uq_{data_type}_excg.yaml"
print(f"Saving excg config to {fname}")
with open(fname, "w") as f:
    yaml.dump(config, f)

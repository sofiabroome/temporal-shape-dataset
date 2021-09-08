import json


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config

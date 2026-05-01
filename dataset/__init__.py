import re

# Define placeholders for dataset paths
VIDEOREWARD = {
    "annotation_path": "./data/train_fixed.json",
    "data_path": "./data",
}

VIDEOREWARD_T = {
    "annotation_path": "./data/train_t.json",
    "data_path": "./data",
}

VIDEOREWARD_POLISHED = {
    "annotation_path": "./data/train_polished_v3.json",
    "data_path": "./data",
}

VIDEOREWARD_T_POLISHED_V3 = {
    "annotation_path": "./data/train_t_polished_v3.json",
    "data_path": "./data",
}

VIDEOREWARD_BOX_T_POLISHED_V3 = {
    "annotation_path": "./data/train_box_t_polished_v3.json",
    "data_path": "./data",
}

VIDEOREWARD_BOX_T_POLISHED_V3_SINGLE = {
    "annotation_path": "./data/train_box_t_polished_v3_single.json",
    "data_path": "./data",
}

VIDEOREWARD_EVAL = {
    "annotation_path": "./data/eval_fixed.json",
    "data_path": "./data",
}

VIDEOREWARD_EVAL_POLISHED_V3 = {
    "annotation_path": "./data/eval_polished_v3.json",
    "data_path": "./data",
}

data_dict = {
    "videoreward": VIDEOREWARD,
    "videoreward_t": VIDEOREWARD_T,
    "videoreward_eval": VIDEOREWARD_EVAL,
    "videoreward_polished": VIDEOREWARD_POLISHED,
    "videoreward_t_polished_v3": VIDEOREWARD_T_POLISHED_V3,
    "videoreward_box_t_polished_v3": VIDEOREWARD_BOX_T_POLISHED_V3,
    "videoreward_box_t_polished_v3_single": VIDEOREWARD_BOX_T_POLISHED_V3_SINGLE,
    "videoreward_eval_polished_v3": VIDEOREWARD_EVAL_POLISHED_V3,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    pass

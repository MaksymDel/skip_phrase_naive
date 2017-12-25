def build_dataset_skip_gram(path_to_data):
    with open(path_to_data, 'r') as f:
        lines = f.readlines()

    return lines

def build_dataset_skip_phrase(path_to_data):
    with open(path_to_data, 'r') as f:
        lines = f.readlines()

    return lines

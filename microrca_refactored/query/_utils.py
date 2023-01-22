from pathlib import Path
import pickle
import csv

def save_as_pickle(path:Path, content: object):
    with open(path, 'wb') as f:
        pickle.dump(content, f)

def load_node_dict(node_dict_path: Path):
    """Load node mapping from csv and parse it into a dict"""
    node_dict = {}
    with open(node_dict_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            node_dict[row[0]] = row[1]
    return node_dict


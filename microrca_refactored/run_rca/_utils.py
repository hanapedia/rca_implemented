from pathlib import Path
import pickle

def load_pickle(path: Path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res

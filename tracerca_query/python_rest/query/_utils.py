import pickle
from pathlib import Path

def recurse_dict_print(d: dict | list, depth=0):
    """Util function to print the schema of a nested dictionaly recursively"""
    if isinstance(d, dict):
        for k, v in d.items():
            print(f'{"  "*depth}{k}: {type(v)}')
            if not (isinstance(v, list) or isinstance(v, dict)): 
                continue
            recurse_dict_print(v, depth=depth+1)
        else:
            return

    elif isinstance(d, list):
        v = next(iter(d), "empty")
        print(f'{"  "*depth}list of: {type(v)}')
        if not (isinstance(v, list) or isinstance(v, dict)): 
            return
        recurse_dict_print(v, depth=depth+1)

    else:
        raise Exception("Input d must be type dict or list")

def load_pickle_file(path: Path):
    """Load Pickle file and return its content"""
    with open(path, 'rb') as f:
        return pickle.load(f)
    

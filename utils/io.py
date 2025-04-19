import json
import os

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

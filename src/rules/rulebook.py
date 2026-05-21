import yaml, os

def load_rulebook(path:str="docs/rulebook.yaml"):
    with open(path,"r") as f:
        return yaml.safe_load(f)

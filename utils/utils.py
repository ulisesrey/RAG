import yaml

def load_config():
    """Load config yaml"""
    with open("config.yaml") as f:
        return yaml.safe_load(f)
import os
import yaml
from abc import ABC, abstractmethod
from llmrouter.data import DataLoader


class MetaRouter(ABC):
    """
    MetaRouter (Base Class)
    -----------------------
    Loads YAML configuration and all data on initialization.
    Subclasses must implement:
        - train()
        - inference()
    """

    def __init__(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 🧭 Compute project root (two levels up from models/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        # Load data via DataLoader
        loader = DataLoader(project_root)
        loader.load_data(config, self)

        # Metric weights
        weights_dict = config.get("metric", {}).get("weights", {})
        self.weights_list = list(weights_dict.values())

        print("✅ MetaRouter initialized successfully (YAML + data loaded).")

    def train(self, *args, **kwargs):
        """Define training behavior for this router."""
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        """Define inference / routing behavior for this router."""
        pass



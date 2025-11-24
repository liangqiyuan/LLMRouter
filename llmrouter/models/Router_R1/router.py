import torch.nn as nn
from llmrouter.models.meta_router import MetaRouter


class RouterR1(MetaRouter):
    """
    Router-R1
    -----------
    Example router that performs R1-like routing.

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route()` method using the underlying `model`
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        """
        Args:
            model (nn.Module):
                Underlying LLM  (e.g., qwen, llama, etc).
            yaml_path (str | None):
                Optional path to YAML config for this router.
            resources (Any, optional):
                Additional shared resources, if needed.
        """
        super().__init__(model=model, yaml_path=yaml_path, resources=resources)

    def route(self, batch):
        """
        Perform routing on a batch of data.
        """
        graph = batch["graph"]
        logits = self.model(graph)
        return {"logits": logits}

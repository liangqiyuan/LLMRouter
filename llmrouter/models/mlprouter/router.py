from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.neural_network import MLPClassifier
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api, generate_task_query, calculate_task_performance


class MLPRouter(MetaRouter):
    """
    MLPRouter
    ----------
    A routing module that leverages a Multi-Layer Perceptron (MLP)
    classifier to select the most suitable language model based on
    query embeddings.

    YAML Configuration Example:
    ---------------------------
    llm_data:
      GPT4:
        size: "175B"
        embedding: [0.12, 0.33, 0.78, 0.44]
      Claude3:
        size: "52B"
        embedding: [0.10, 0.25, 0.70, 0.50]
    optional:
      mlp_model_path: "configs/mlp_model.pkl"
      hidden_layer_sizes: [64, 32]
      activation: "relu"
      solver: "adam"
      max_iter: 300
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the MLPRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Builds an MLP classifier with the specified hyperparameters.
            3. Prepares the training embeddings and corresponding model labels.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        mlp_params = self.cfg["hparam"]
        self.mlp_model = MLPClassifier(**mlp_params)

        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        query_embedding_id = routing_best["embedding_id"].tolist()
        self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
        self.model_name_list = routing_best["model_name"].tolist()

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the trained MLP model.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_model_path)

        query_embedding = [get_longformer_embedding(query["query"]).numpy()]
        model_name = self.mlp_model.predict(query_embedding)[0]

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the trained MLP model and execute them.

        This method performs end-to-end processing for each query:
        1. Routes the query to get the best model
        2. Applies task-specific prompt formatting if task_name is provided
        3. Calls the routed model via API to get response
        4. Calculates performance metrics if ground truth is available

        Args:
            batch (Any, optional):
                If provided, routes the provided batch. If None, uses self.query_data_test from loaded data.
            task_name (str, optional):
                Task name for prompt formatting (e.g., "mmlu", "gsm8k", "commonsense_qa").

        Returns:
            list of dict:
                A list of query dictionaries with response, tokens, and performance metrics.
        """
        # Load model once
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_model_path)

        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
            else:
                print("Warning: No batch provided and no test data available for batch routing.")
                return []

        query_data_output = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                row_copy = copy.copy(row)
                original_query = row_copy.get("query", "")
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)
                row_task_name = task_name

            # Step 1: Route the query
            query_embedding = [get_longformer_embedding(original_query).numpy()]
            model_name = self.mlp_model.predict(query_embedding)[0]
            row_copy["model_name"] = model_name

            # Step 2: Format query if task_name is provided
            if row_task_name:
                try:
                    sample_data = {
                        "query": original_query,
                        "choices": row_copy.get("choices", None) if isinstance(row_copy, dict) else None
                    }
                    formatted_query = generate_task_query(row_task_name, sample_data)
                    row_copy["formatted_query"] = formatted_query
                    query_text_for_execution = formatted_query
                except (ValueError, KeyError) as e:
                    print(f"Warning: Failed to format query with task '{row_task_name}': {e}. Using original query.")
                    query_text_for_execution = original_query
            else:
                query_text_for_execution = original_query

            # Step 3: Call API to get response
            # Get API endpoint and model name from llm_data if available
            api_model_name = model_name
            api_endpoint = None
            if hasattr(self, 'llm_data') and self.llm_data and model_name in self.llm_data:
                api_model_name = self.llm_data[model_name].get("model", model_name)
                # Get API endpoint from llm_data, fallback to router config
                api_endpoint = self.llm_data[model_name].get(
                    "api_endpoint",
                    self.cfg.get("api_endpoint")
                )
            
            # If still no endpoint found, try router config
            if api_endpoint is None:
                api_endpoint = self.cfg.get("api_endpoint")
            
            # Validate that we have an endpoint
            if not api_endpoint:
                raise ValueError(
                    f"API endpoint not found for model '{model_name}'. "
                    f"Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config."
                )

            request = {
                "api_endpoint": api_endpoint,
                "query": query_text_for_execution,
                "model_name": model_name,
                "api_name": api_model_name
            }

            try:
                result = call_api(request, max_tokens=1024, temperature=0.7)
                response = result.get("response", "")
                prompt_tokens = result.get("prompt_tokens", 0)
                completion_tokens = result.get("completion_tokens", 0)
                success = "error" not in result
            except Exception as e:
                print(f"Error calling API for query: {e}")
                response = ""
                prompt_tokens = 0
                completion_tokens = 0
                success = False

            row_copy["response"] = response
            row_copy["prompt_tokens"] = prompt_tokens
            row_copy["completion_tokens"] = completion_tokens
            row_copy["input_token"] = prompt_tokens
            row_copy["output_token"] = completion_tokens
            row_copy["success"] = success

            # Step 4: Calculate task performance if ground truth is available
            ground_truth = row_copy.get("ground_truth") or row_copy.get("gt") or row_copy.get("answer")
            metric = row_copy.get("metric")
            if ground_truth:
                task_performance = calculate_task_performance(
                    prediction=response,
                    ground_truth=ground_truth,
                    task_name=row_task_name,
                    metric=metric
                )
                if task_performance is not None:
                    row_copy["task_performance"] = task_performance

            query_data_output.append(row_copy)

        return query_data_output


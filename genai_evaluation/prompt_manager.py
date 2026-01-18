"""
Prompt Manager for Unity Catalog

Manages prompt templates with versioning, governance, and retrieval using Unity Catalog.
Supports prompt storage as functions and metadata tracking with MLflow.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class PromptTemplate:
    """Represents a versioned prompt template"""
    name: str
    template: str
    version: str
    description: str
    parameters: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    tags: Dict[str, str]


class PromptManager:
    """
    Manages prompts in Unity Catalog with versioning and governance.

    This class provides methods to:
    - Register prompts as versioned artifacts in Unity Catalog
    - Retrieve prompts by name and version
    - Track prompt usage with MLflow
    - Manage prompt metadata and tags

    Example:
        >>> pm = PromptManager(catalog="main", schema="prompts")
        >>> pm.register_prompt(
        ...     name="rag_qa_prompt",
        ...     template="Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:",
        ...     description="RAG QA prompt for answering questions",
        ...     parameters=["context", "question"],
        ...     tags={"use_case": "qa", "model": "llama3"}
        ... )
    """

    def __init__(
        self,
        catalog: str,
        schema: str,
        mlflow_experiment: Optional[str] = None
    ):
        """
        Initialize the PromptManager.

        Args:
            catalog: Unity Catalog catalog name
            schema: Unity Catalog schema name
            mlflow_experiment: MLflow experiment name for tracking (optional)
        """
        self.catalog = catalog
        self.schema = schema
        self.namespace = f"{catalog}.{schema}"
        self.client = MlflowClient()

        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

    def register_prompt(
        self,
        name: str,
        template: str,
        description: str,
        parameters: List[str],
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> PromptTemplate:
        """
        Register a new prompt template in Unity Catalog.

        Args:
            name: Unique name for the prompt
            template: The prompt template string with parameter placeholders
            description: Description of the prompt's purpose
            parameters: List of parameter names used in the template
            tags: Optional tags for categorization
            metadata: Optional metadata (model info, use case, etc.)
            version: Optional version string (auto-generated if not provided)

        Returns:
            PromptTemplate object with registered details
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        tags = tags or {}
        metadata = metadata or {}

        # Log prompt to MLflow
        with mlflow.start_run(run_name=f"register_prompt_{name}_{version}"):
            # Log prompt template as artifact
            mlflow.log_text(template, f"prompts/{name}_v{version}.txt")

            # Log parameters and metadata
            mlflow.log_params({
                "prompt_name": name,
                "version": version,
                "num_parameters": len(parameters)
            })

            mlflow.log_dict(
                {
                    "template": template,
                    "description": description,
                    "parameters": parameters,
                    "metadata": metadata
                },
                f"prompts/{name}_v{version}_metadata.json"
            )

            # Log tags
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            mlflow.set_tag("prompt_name", name)
            mlflow.set_tag("prompt_version", version)

            run_id = mlflow.active_run().info.run_id

        prompt_template = PromptTemplate(
            name=name,
            template=template,
            version=version,
            description=description,
            parameters=parameters,
            metadata={**metadata, "mlflow_run_id": run_id},
            created_at=datetime.now(),
            tags=tags
        )

        return prompt_template

    def get_prompt(
        self,
        name: str,
        version: Optional[str] = "latest"
    ) -> Optional[PromptTemplate]:
        """
        Retrieve a prompt template by name and version.

        Args:
            name: The prompt name
            version: The version to retrieve (default: "latest")

        Returns:
            PromptTemplate if found, None otherwise
        """
        # Search for runs with matching prompt name
        experiment = mlflow.get_experiment_by_name(mlflow.get_experiment_by_name)
        if not experiment:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.prompt_name = '{name}'",
            order_by=["start_time DESC"],
            max_results=1 if version == "latest" else 100
        )

        if runs.empty:
            return None

        if version != "latest":
            runs = runs[runs["tags.prompt_version"] == version]
            if runs.empty:
                return None

        run = runs.iloc[0]
        run_id = run["run_id"]

        # Download prompt metadata
        client = MlflowClient()
        local_path = client.download_artifacts(
            run_id,
            f"prompts/{name}_v{run['tags.prompt_version']}_metadata.json"
        )

        import json
        with open(local_path, 'r') as f:
            prompt_data = json.load(f)

        return PromptTemplate(
            name=name,
            template=prompt_data["template"],
            version=run["tags.prompt_version"],
            description=prompt_data["description"],
            parameters=prompt_data["parameters"],
            metadata=prompt_data.get("metadata", {}),
            created_at=datetime.fromisoformat(run["start_time"]),
            tags={k.replace("tags.", ""): v for k, v in run.items() if k.startswith("tags.")}
        )

    def list_prompts(
        self,
        tag_filter: Optional[Dict[str, str]] = None
    ) -> List[PromptTemplate]:
        """
        List all registered prompts, optionally filtered by tags.

        Args:
            tag_filter: Optional dictionary of tag key-value pairs to filter by

        Returns:
            List of PromptTemplate objects
        """
        experiment = mlflow.get_experiment_by_name(mlflow.active_experiment().name)
        if not experiment:
            return []

        filter_string = "tags.prompt_name != ''"
        if tag_filter:
            for key, value in tag_filter.items():
                filter_string += f" and tags.{key} = '{value}'"

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )

        prompts = []
        for _, run in runs.iterrows():
            if "tags.prompt_name" in run:
                prompt_name = run["tags.prompt_name"]
                prompt_version = run.get("tags.prompt_version", "unknown")

                prompt = self.get_prompt(prompt_name, prompt_version)
                if prompt:
                    prompts.append(prompt)

        return prompts

    def format_prompt(
        self,
        prompt_template: PromptTemplate,
        **kwargs
    ) -> str:
        """
        Format a prompt template with provided parameters.

        Args:
            prompt_template: The PromptTemplate to format
            **kwargs: Parameter values to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If required parameters are missing
        """
        missing_params = set(prompt_template.parameters) - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        return prompt_template.template.format(**kwargs)

    def compare_prompts(
        self,
        prompt_names: List[str],
        test_inputs: List[Dict[str, Any]],
        evaluation_fn: callable
    ) -> Dict[str, Any]:
        """
        Compare multiple prompt versions on test inputs.

        Args:
            prompt_names: List of prompt names or name:version pairs
            test_inputs: List of input dictionaries for testing
            evaluation_fn: Function to evaluate prompt outputs

        Returns:
            Comparison results with metrics for each prompt
        """
        results = {}

        for prompt_name in prompt_names:
            if ":" in prompt_name:
                name, version = prompt_name.split(":")
            else:
                name, version = prompt_name, "latest"

            prompt = self.get_prompt(name, version)
            if not prompt:
                continue

            prompt_results = []
            for test_input in test_inputs:
                formatted = self.format_prompt(prompt, **test_input)
                score = evaluation_fn(formatted, test_input)
                prompt_results.append(score)

            results[f"{name}:{version}"] = {
                "scores": prompt_results,
                "avg_score": sum(prompt_results) / len(prompt_results)
            }

        return results

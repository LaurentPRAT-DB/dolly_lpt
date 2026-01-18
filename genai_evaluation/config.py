"""
Configuration for GenAI Evaluation Framework

Centralized configuration for prompt management, evaluation, and model settings.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class UnityConfig:
    """Unity Catalog configuration"""
    catalog: str = "main"
    schema: str = "prompts"
    volume: Optional[str] = None

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"


@dataclass
class MLflowConfig:
    """MLflow tracking configuration"""
    experiment_base_path: str = "/Users"
    prompt_experiment_name: str = "rag_prompts"
    evaluation_experiment_name: str = "rag_evaluation"
    tracking_uri: Optional[str] = None

    @property
    def prompt_experiment_path(self) -> str:
        return f"{self.experiment_base_path}/{self.prompt_experiment_name}"

    @property
    def evaluation_experiment_path(self) -> str:
        return f"{self.experiment_base_path}/{self.evaluation_experiment_name}"


@dataclass
class JudgeModelConfig:
    """LLM Judge model configuration"""
    # Available Databricks Foundation Models
    model_name: str = "databricks-meta-llama-3-70b-instruct"
    endpoint_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 500
    top_p: float = 1.0

    # Alternative models
    LLAMA_3_70B = "databricks-meta-llama-3-70b-instruct"
    LLAMA_3_8B = "databricks-meta-llama-3-8b-instruct"
    MIXTRAL_8X7B = "databricks-mixtral-8x7b-instruct"
    DBRX_INSTRUCT = "databricks-dbrx-instruct"


@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration"""
    parallel_workers: int = 4
    batch_size: int = 10
    retry_attempts: int = 3
    timeout_seconds: int = 60
    cache_results: bool = True

    # Evaluation thresholds
    faithfulness_threshold: float = 0.7
    relevance_threshold: float = 0.7
    groundedness_threshold: float = 0.8
    completeness_threshold: float = 0.6


@dataclass
class RAGConfig:
    """RAG-specific configuration"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    rerank: bool = True
    context_window: int = 4096


class Config:
    """
    Main configuration class combining all settings.

    Usage:
        >>> config = Config()
        >>> config.unity.catalog = "production"
        >>> config.judge.model_name = Config.judge.LLAMA_3_8B
    """

    def __init__(
        self,
        unity: Optional[UnityConfig] = None,
        mlflow: Optional[MLflowConfig] = None,
        judge: Optional[JudgeModelConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        rag: Optional[RAGConfig] = None
    ):
        self.unity = unity or UnityConfig()
        self.mlflow = mlflow or MLflowConfig()
        self.judge = judge or JudgeModelConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.rag = rag or RAGConfig()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Create Config from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            Config instance
        """
        return cls(
            unity=UnityConfig(**config_dict.get("unity", {})),
            mlflow=MLflowConfig(**config_dict.get("mlflow", {})),
            judge=JudgeModelConfig(**config_dict.get("judge", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            rag=RAGConfig(**config_dict.get("rag", {}))
        )

    def to_dict(self) -> dict:
        """
        Convert Config to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "unity": self.unity.__dict__,
            "mlflow": self.mlflow.__dict__,
            "judge": self.judge.__dict__,
            "evaluation": self.evaluation.__dict__,
            "rag": self.rag.__dict__
        }


# Default configuration instance
default_config = Config()

"""
RAG Evaluation Framework with LLM-as-Judge

Provides evaluation metrics for RAG applications using foundation models as judges.
Supports multiple evaluation dimensions: faithfulness, relevance, groundedness, etc.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import mlflow
from concurrent.futures import ThreadPoolExecutor, as_completed


class RAGMetric(Enum):
    """Supported RAG evaluation metrics"""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RELEVANCE = "context_relevance"
    GROUNDEDNESS = "groundedness"
    COMPLETENESS = "completeness"


@dataclass
class EvaluationResult:
    """Results from a single evaluation"""
    metric: RAGMetric
    score: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class RAGExample:
    """A single RAG example for evaluation"""
    question: str
    context: str
    answer: str
    reference_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMJudge:
    """
    LLM-based evaluator that uses foundation models to judge RAG outputs.

    This class uses Databricks Foundation Model APIs to evaluate RAG responses
    across multiple dimensions. It supports various metrics and can be configured
    to use different judge models.

    Example:
        >>> judge = LLMJudge(
        ...     model_name="databricks-meta-llama-3-70b-instruct",
        ...     endpoint_name="llama3-70b-endpoint"
        ... )
        >>> result = judge.evaluate_faithfulness(
        ...     question="What is the capital of France?",
        ...     context="Paris is the capital and largest city of France.",
        ...     answer="The capital of France is Paris."
        ... )
    """

    # Evaluation prompt templates
    FAITHFULNESS_PROMPT = """You are an expert evaluator. Assess whether the answer is faithful to the provided context.

Question: {question}

Context: {context}

Answer: {answer}

Evaluate if the answer is supported by the context. The answer should not include information that contradicts or is not present in the context.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1, where 1 is fully faithful>,
    "reasoning": "<brief explanation of your assessment>"
}}

Response:"""

    ANSWER_RELEVANCE_PROMPT = """You are an expert evaluator. Assess whether the answer is relevant to the question.

Question: {question}

Answer: {answer}

Evaluate if the answer directly addresses the question. The answer should be on-topic and provide information that helps answer the question.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1, where 1 is highly relevant>,
    "reasoning": "<brief explanation of your assessment>"
}}

Response:"""

    CONTEXT_RELEVANCE_PROMPT = """You are an expert evaluator. Assess whether the context is relevant to the question.

Question: {question}

Context: {context}

Evaluate if the context contains information that could help answer the question. The context should be on-topic and related to the question.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1, where 1 is highly relevant>,
    "reasoning": "<brief explanation of your assessment>"
}}

Response:"""

    GROUNDEDNESS_PROMPT = """You are an expert evaluator. Assess whether the answer is grounded in the provided context.

Context: {context}

Answer: {answer}

Evaluate if every claim in the answer can be traced back to the context. The answer should not contain unsupported claims.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1, where 1 is fully grounded>,
    "reasoning": "<brief explanation of your assessment>"
}}

Response:"""

    COMPLETENESS_PROMPT = """You are an expert evaluator. Assess whether the answer is complete relative to the question and context.

Question: {question}

Context: {context}

Answer: {answer}

Evaluate if the answer provides a complete response to the question given the available context. The answer should cover key points without omitting important information.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1, where 1 is fully complete>,
    "reasoning": "<brief explanation of your assessment>"
}}

Response:"""

    def __init__(
        self,
        model_name: str = "databricks-meta-llama-3-70b-instruct",
        endpoint_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize the LLM Judge.

        Args:
            model_name: Name of the foundation model to use as judge
            endpoint_name: Optional endpoint name for model serving
            temperature: Temperature for generation (0 for deterministic)
            max_tokens: Maximum tokens for judge response
        """
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_model(self, prompt: str) -> str:
        """
        Call the foundation model API.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Model response text
        """
        # This is a placeholder for actual Databricks Foundation Model API call
        # In production, use the Databricks SDK or REST API
        try:
            import openai
            from databricks.sdk import WorkspaceClient

            # For Databricks Foundation Model APIs
            # w = WorkspaceClient()
            # response = w.serving_endpoints.query(
            #     name=self.endpoint_name or self.model_name,
            #     inputs={"prompt": prompt},
            #     max_tokens=self.max_tokens,
            #     temperature=self.temperature
            # )
            # return response.predictions[0]["candidates"][0]["text"]

            # Fallback: Log the prompt for demonstration
            print(f"[LLM Judge] Calling {self.model_name}")
            print(f"[LLM Judge] Prompt: {prompt[:200]}...")

            # Return mock response for demonstration
            return '{"score": 0.85, "reasoning": "The answer is well-supported by the context."}'

        except Exception as e:
            raise RuntimeError(f"Error calling model {self.model_name}: {e}")

    def _parse_response(self, response: str) -> Tuple[float, str]:
        """
        Parse the JSON response from the judge model.

        Args:
            response: Raw response from the model

        Returns:
            Tuple of (score, reasoning)
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0]
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response)
            return result["score"], result["reasoning"]
        except Exception as e:
            # Fallback: try to extract score and reasoning
            print(f"Warning: Failed to parse response: {e}")
            return 0.5, f"Failed to parse: {response}"

    def evaluate_faithfulness(
        self,
        question: str,
        context: str,
        answer: str
    ) -> EvaluationResult:
        """
        Evaluate if the answer is faithful to the context.

        Args:
            question: The question asked
            context: Retrieved context
            answer: Generated answer

        Returns:
            EvaluationResult with faithfulness score
        """
        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question,
            context=context,
            answer=answer
        )

        response = self._call_model(prompt)
        score, reasoning = self._parse_response(response)

        return EvaluationResult(
            metric=RAGMetric.FAITHFULNESS,
            score=score,
            reasoning=reasoning,
            metadata={"question": question, "context_length": len(context)}
        )

    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> EvaluationResult:
        """
        Evaluate if the answer is relevant to the question.

        Args:
            question: The question asked
            answer: Generated answer

        Returns:
            EvaluationResult with relevance score
        """
        prompt = self.ANSWER_RELEVANCE_PROMPT.format(
            question=question,
            answer=answer
        )

        response = self._call_model(prompt)
        score, reasoning = self._parse_response(response)

        return EvaluationResult(
            metric=RAGMetric.ANSWER_RELEVANCE,
            score=score,
            reasoning=reasoning,
            metadata={"question": question}
        )

    def evaluate_context_relevance(
        self,
        question: str,
        context: str
    ) -> EvaluationResult:
        """
        Evaluate if the context is relevant to the question.

        Args:
            question: The question asked
            context: Retrieved context

        Returns:
            EvaluationResult with context relevance score
        """
        prompt = self.CONTEXT_RELEVANCE_PROMPT.format(
            question=question,
            context=context
        )

        response = self._call_model(prompt)
        score, reasoning = self._parse_response(response)

        return EvaluationResult(
            metric=RAGMetric.CONTEXT_RELEVANCE,
            score=score,
            reasoning=reasoning,
            metadata={"question": question, "context_length": len(context)}
        )

    def evaluate_groundedness(
        self,
        context: str,
        answer: str
    ) -> EvaluationResult:
        """
        Evaluate if the answer is grounded in the context.

        Args:
            context: Retrieved context
            answer: Generated answer

        Returns:
            EvaluationResult with groundedness score
        """
        prompt = self.GROUNDEDNESS_PROMPT.format(
            context=context,
            answer=answer
        )

        response = self._call_model(prompt)
        score, reasoning = self._parse_response(response)

        return EvaluationResult(
            metric=RAGMetric.GROUNDEDNESS,
            score=score,
            reasoning=reasoning,
            metadata={"context_length": len(context)}
        )

    def evaluate_completeness(
        self,
        question: str,
        context: str,
        answer: str
    ) -> EvaluationResult:
        """
        Evaluate if the answer is complete.

        Args:
            question: The question asked
            context: Retrieved context
            answer: Generated answer

        Returns:
            EvaluationResult with completeness score
        """
        prompt = self.COMPLETENESS_PROMPT.format(
            question=question,
            context=context,
            answer=answer
        )

        response = self._call_model(prompt)
        score, reasoning = self._parse_response(response)

        return EvaluationResult(
            metric=RAGMetric.COMPLETENESS,
            score=score,
            reasoning=reasoning,
            metadata={"question": question, "context_length": len(context)}
        )

    def evaluate_all(
        self,
        example: RAGExample,
        metrics: Optional[List[RAGMetric]] = None
    ) -> Dict[RAGMetric, EvaluationResult]:
        """
        Evaluate a RAG example across multiple metrics.

        Args:
            example: RAGExample to evaluate
            metrics: List of metrics to evaluate (default: all)

        Returns:
            Dictionary mapping metrics to evaluation results
        """
        if metrics is None:
            metrics = list(RAGMetric)

        results = {}

        for metric in metrics:
            if metric == RAGMetric.FAITHFULNESS:
                results[metric] = self.evaluate_faithfulness(
                    example.question, example.context, example.answer
                )
            elif metric == RAGMetric.ANSWER_RELEVANCE:
                results[metric] = self.evaluate_answer_relevance(
                    example.question, example.answer
                )
            elif metric == RAGMetric.CONTEXT_RELEVANCE:
                results[metric] = self.evaluate_context_relevance(
                    example.question, example.context
                )
            elif metric == RAGMetric.GROUNDEDNESS:
                results[metric] = self.evaluate_groundedness(
                    example.context, example.answer
                )
            elif metric == RAGMetric.COMPLETENESS:
                results[metric] = self.evaluate_completeness(
                    example.question, example.context, example.answer
                )

        return results


class RAGEvaluationPipeline:
    """
    End-to-end evaluation pipeline for RAG applications.

    Coordinates evaluation across multiple examples, tracks results with MLflow,
    and provides aggregate metrics and analysis.

    Example:
        >>> pipeline = RAGEvaluationPipeline(
        ...     judge=LLMJudge(model_name="databricks-meta-llama-3-70b-instruct"),
        ...     mlflow_experiment="rag_evaluation"
        ... )
        >>> results = pipeline.evaluate_dataset(examples, metrics=[RAGMetric.FAITHFULNESS])
    """

    def __init__(
        self,
        judge: LLMJudge,
        mlflow_experiment: Optional[str] = None,
        parallel_workers: int = 4
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            judge: LLMJudge instance to use for evaluation
            mlflow_experiment: MLflow experiment name for tracking
            parallel_workers: Number of parallel workers for evaluation
        """
        self.judge = judge
        self.parallel_workers = parallel_workers

        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

    def evaluate_dataset(
        self,
        examples: List[RAGExample],
        metrics: Optional[List[RAGMetric]] = None,
        log_to_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset of RAG examples.

        Args:
            examples: List of RAGExample objects to evaluate
            metrics: List of metrics to compute (default: all)
            log_to_mlflow: Whether to log results to MLflow

        Returns:
            Dictionary with aggregate results and per-example scores
        """
        if metrics is None:
            metrics = list(RAGMetric)

        all_results = []

        if log_to_mlflow:
            run_name = f"rag_eval_{len(examples)}_examples"
            mlflow.start_run(run_name=run_name)

        # Evaluate examples in parallel
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self.judge.evaluate_all, example, metrics): i
                for i, example in enumerate(examples)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    all_results.append({
                        "example_idx": idx,
                        "example": examples[idx],
                        "results": result
                    })
                except Exception as e:
                    print(f"Error evaluating example {idx}: {e}")

        # Compute aggregate metrics
        aggregate_scores = {}
        for metric in metrics:
            scores = [
                r["results"][metric].score
                for r in all_results
                if metric in r["results"]
            ]
            if scores:
                aggregate_scores[metric.value] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }

        # Log to MLflow
        if log_to_mlflow:
            mlflow.log_params({
                "num_examples": len(examples),
                "judge_model": self.judge.model_name,
                "metrics": [m.value for m in metrics]
            })

            for metric_name, scores in aggregate_scores.items():
                mlflow.log_metrics({
                    f"{metric_name}_mean": scores["mean"],
                    f"{metric_name}_min": scores["min"],
                    f"{metric_name}_max": scores["max"]
                })

            # Log detailed results as artifact
            mlflow.log_dict(
                {
                    "aggregate": aggregate_scores,
                    "detailed_results": [
                        {
                            "example_idx": r["example_idx"],
                            "question": r["example"].question,
                            "scores": {
                                m.value: {
                                    "score": r["results"][m].score,
                                    "reasoning": r["results"][m].reasoning
                                }
                                for m in metrics if m in r["results"]
                            }
                        }
                        for r in all_results
                    ]
                },
                "evaluation_results.json"
            )

            mlflow.end_run()

        return {
            "aggregate_scores": aggregate_scores,
            "detailed_results": all_results,
            "num_examples": len(examples)
        }

    def compare_systems(
        self,
        system_results: Dict[str, List[RAGExample]],
        metrics: Optional[List[RAGMetric]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple RAG systems on the same questions.

        Args:
            system_results: Dict mapping system names to their RAG examples
            metrics: List of metrics to compute (default: all)

        Returns:
            Comparison results across systems
        """
        comparison = {}

        with mlflow.start_run(run_name="system_comparison"):
            for system_name, examples in system_results.items():
                print(f"Evaluating system: {system_name}")
                results = self.evaluate_dataset(
                    examples,
                    metrics=metrics,
                    log_to_mlflow=False
                )
                comparison[system_name] = results["aggregate_scores"]

                # Log system-specific metrics
                for metric_name, scores in results["aggregate_scores"].items():
                    mlflow.log_metric(f"{system_name}_{metric_name}_mean", scores["mean"])

            mlflow.log_dict(comparison, "system_comparison.json")

        return comparison

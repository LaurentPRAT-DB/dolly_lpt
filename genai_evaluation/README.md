# GenAI Evaluation Framework

A comprehensive framework for managing prompts and evaluating RAG applications on Databricks using LLM-as-judge patterns.

## Features

### Prompt Management
- **Unity Catalog Integration**: Store and version prompts with full governance
- **Metadata Tracking**: Track prompt performance, authors, and usage
- **Version Control**: Semantic versioning for prompt templates
- **MLflow Integration**: Experiment tracking and artifact storage
- **Tag-based Organization**: Categorize and filter prompts by use case

### RAG Evaluation
- **LLM-as-Judge**: Use foundation models to evaluate RAG outputs
- **Multiple Metrics**: Faithfulness, relevance, groundedness, completeness
- **Parallel Processing**: Efficient batch evaluation with threading
- **MLflow Tracking**: Automatic logging of evaluation results
- **Comparison Tools**: A/B test different prompt versions

## Quick Start

### Installation

```python
# Install required dependencies
%pip install mlflow>=2.9.0 databricks-sdk>=0.18.0
```

### Basic Usage

#### 1. Prompt Management

```python
from genai_evaluation.prompt_manager import PromptManager

# Initialize
pm = PromptManager(
    catalog="main",
    schema="prompts",
    mlflow_experiment="/Users/your-email/rag_prompts"
)

# Register a prompt
prompt = pm.register_prompt(
    name="rag_qa_basic",
    template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
    description="Basic RAG QA prompt",
    parameters=["context", "question"],
    tags={"use_case": "qa", "model": "llama3"}
)

# Retrieve a prompt
retrieved = pm.get_prompt("rag_qa_basic", version="latest")

# Format with parameters
formatted = pm.format_prompt(
    retrieved,
    context="Paris is the capital of France.",
    question="What is the capital of France?"
)
```

#### 2. RAG Evaluation

```python
from genai_evaluation.rag_evaluator import (
    LLMJudge, RAGEvaluationPipeline, RAGExample, RAGMetric
)

# Initialize judge
judge = LLMJudge(
    model_name="databricks-meta-llama-3-70b-instruct",
    temperature=0.0
)

# Create test example
example = RAGExample(
    question="What is the capital of France?",
    context="Paris is the capital of France.",
    answer="The capital of France is Paris."
)

# Evaluate
results = judge.evaluate_all(example)
for metric, result in results.items():
    print(f"{metric.value}: {result.score:.2f}")
```

#### 3. Full Pipeline

```python
# Initialize pipeline
pipeline = RAGEvaluationPipeline(
    judge=judge,
    mlflow_experiment="/Users/your-email/rag_evaluation",
    parallel_workers=4
)

# Evaluate dataset
results = pipeline.evaluate_dataset(
    examples=test_examples,
    metrics=[RAGMetric.FAITHFULNESS, RAGMetric.ANSWER_RELEVANCE],
    log_to_mlflow=True
)

print(f"Mean faithfulness: {results['aggregate_scores']['faithfulness']['mean']:.3f}")
```

## Architecture

```
genai_evaluation/
├── __init__.py              # Package initialization
├── prompt_manager.py        # Prompt management with Unity Catalog
├── rag_evaluator.py        # LLM-as-judge evaluation framework
├── config.py               # Configuration management
└── README.md               # This file

examples/
└── prompt_management_evaluation_example.py  # Complete walkthrough
```

## Evaluation Metrics

### Faithfulness
Evaluates whether the answer is faithful to the provided context. Checks if the answer contains information not present in or contradicting the context.

**Score Range**: 0.0 - 1.0 (higher is better)

### Answer Relevance
Assesses whether the answer directly addresses the question. The answer should be on-topic and provide useful information.

**Score Range**: 0.0 - 1.0 (higher is better)

### Context Relevance
Evaluates whether the retrieved context is relevant to the question. Good context should contain information that helps answer the question.

**Score Range**: 0.0 - 1.0 (higher is better)

### Groundedness
Checks if every claim in the answer can be traced back to the context. Similar to faithfulness but more strict.

**Score Range**: 0.0 - 1.0 (higher is better)

### Completeness
Assesses whether the answer provides a complete response given the question and available context.

**Score Range**: 0.0 - 1.0 (higher is better)

## Configuration

Customize settings using the `Config` class:

```python
from genai_evaluation.config import Config

config = Config()
config.unity.catalog = "production"
config.unity.schema = "rag_prompts"
config.judge.model_name = "databricks-meta-llama-3-70b-instruct"
config.evaluation.parallel_workers = 8
config.evaluation.faithfulness_threshold = 0.8
```

## Best Practices

### Prompt Management
1. **Version Control**: Always version your prompts with descriptive tags
2. **Metadata**: Track author, performance metrics, and use cases
3. **Testing**: Test prompts on diverse examples before production
4. **Documentation**: Document prompt parameters and expected behavior

### Evaluation
1. **Representative Data**: Use diverse, real-world examples for evaluation
2. **Multiple Metrics**: Don't rely on a single metric
3. **Parallel Processing**: Use parallel workers for large datasets
4. **Thresholds**: Set appropriate quality thresholds for your use case
5. **Continuous Monitoring**: Regularly evaluate production prompts

### Model Selection
- **Llama 3 70B**: Best balance of quality and cost for evaluation
- **Llama 3 8B**: Faster, more cost-effective for simpler evaluations
- **DBRX Instruct**: Specialized for complex reasoning tasks
- **Mixtral 8x7B**: Good alternative with strong performance

## Examples

See `/examples/prompt_management_evaluation_example.py` for a complete walkthrough including:
- Registering multiple prompt versions
- Evaluating RAG outputs across all metrics
- Comparing prompt performance
- A/B testing different templates
- Custom evaluation functions
- MLflow integration

## Integration with Production Systems

### Model Serving Endpoint

```python
# Deploy evaluated prompt as serving endpoint
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get best performing prompt
best_prompt = pm.get_prompt("rag_qa_production", version="latest")

# Use in serving endpoint prediction
response = w.serving_endpoints.query(
    name="rag-endpoint",
    inputs={
        "prompt": pm.format_prompt(
            best_prompt,
            context=retrieved_context,
            question=user_question
        )
    }
)
```

### Continuous Evaluation

```python
# Set up scheduled evaluation job
from databricks.sdk.service.jobs import Task, NotebookTask

# Create evaluation job that runs daily
evaluation_job = w.jobs.create(
    name="rag_prompt_evaluation",
    tasks=[
        Task(
            task_key="evaluate_prompts",
            notebook_task=NotebookTask(
                notebook_path="/Repos/your-org/dolly/examples/prompt_management_evaluation_example"
            )
        )
    ],
    schedule={"quartz_cron_expression": "0 0 1 * * ?"}  # Daily at 1 AM
)
```

## Troubleshooting

### Common Issues

1. **Model API Errors**: Ensure you have access to Foundation Model APIs in your workspace
2. **Unity Catalog Permissions**: Verify you have CREATE/READ permissions on the catalog and schema
3. **MLflow Tracking**: Check that your experiment path exists and you have write access
4. **Token Limits**: For long contexts, adjust `max_tokens` in `JudgeModelConfig`

### Performance Optimization

- Use parallel workers (4-8) for large evaluation datasets
- Cache evaluation results to avoid redundant API calls
- Batch examples when possible
- Use smaller judge models (8B) for simple evaluations

## Contributing

To extend the framework:

1. Add new evaluation metrics in `rag_evaluator.py`
2. Implement custom prompt templates
3. Create domain-specific evaluation functions
4. Add support for additional foundation models

## Support

For issues or questions:
- GitHub Issues: [databrickslabs/dolly/issues](https://github.com/databrickslabs/dolly/issues)
- Databricks Community: [community.databricks.com](https://community.databricks.com)
- Documentation: [docs.databricks.com/generative-ai](https://docs.databricks.com/generative-ai)

## License

Apache License 2.0 - see LICENSE file for details

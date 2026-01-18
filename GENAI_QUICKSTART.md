# GenAI Evaluation Framework - Quick Start Guide

Get started with prompt management and RAG evaluation on Databricks in 5 minutes.

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to Databricks Foundation Model APIs
- DBR 14.3 LTS ML or higher

## Step 1: Install Dependencies

```python
# In a Databricks notebook
%pip install mlflow>=2.9.0 databricks-sdk>=0.18.0
dbutils.library.restartPython()
```

## Step 2: Set Up Your Environment

```python
# Add the dolly repo to your Python path
import sys
sys.path.append("/Workspace/Repos/your-username/dolly")

from genai_evaluation.prompt_manager import PromptManager
from genai_evaluation.rag_evaluator import LLMJudge, RAGExample, RAGMetric
```

## Step 3: Register Your First Prompt

```python
# Initialize Prompt Manager
pm = PromptManager(
    catalog="main",
    schema="prompts",
    mlflow_experiment="/Users/your-email@company.com/rag_prompts"
)

# Register a RAG prompt
prompt = pm.register_prompt(
    name="my_rag_prompt",
    template="""Context: {context}

Question: {question}

Answer:""",
    description="My first RAG prompt",
    parameters=["context", "question"],
    tags={"version": "v1", "use_case": "qa"}
)

print(f"✓ Registered prompt: {prompt.name} v{prompt.version}")
```

## Step 4: Evaluate RAG Outputs

```python
# Initialize LLM Judge
judge = LLMJudge(
    model_name="databricks-meta-llama-3-70b-instruct",
    temperature=0.0
)

# Create a test example
example = RAGExample(
    question="What is machine learning?",
    context="Machine learning is a branch of AI that enables systems to learn from data.",
    answer="Machine learning is an AI technique that allows systems to learn from data automatically."
)

# Evaluate
result = judge.evaluate_faithfulness(
    question=example.question,
    context=example.context,
    answer=example.answer
)

print(f"✓ Faithfulness Score: {result.score:.2f}")
print(f"  Reasoning: {result.reasoning}")
```

## Step 5: Run Full Evaluation

```python
from genai_evaluation.rag_evaluator import RAGEvaluationPipeline

# Create multiple test examples
test_examples = [
    RAGExample(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        answer="The capital of France is Paris."
    ),
    RAGExample(
        question="Who invented the telephone?",
        context="Alexander Graham Bell is credited with inventing the telephone in 1876.",
        answer="Alexander Graham Bell invented the telephone in 1876."
    )
]

# Initialize pipeline
pipeline = RAGEvaluationPipeline(
    judge=judge,
    mlflow_experiment="/Users/your-email@company.com/rag_evaluation"
)

# Run evaluation
results = pipeline.evaluate_dataset(
    examples=test_examples,
    metrics=[RAGMetric.FAITHFULNESS, RAGMetric.ANSWER_RELEVANCE],
    log_to_mlflow=True
)

# View results
print("\n✓ Evaluation Complete!")
for metric, scores in results["aggregate_scores"].items():
    print(f"  {metric}: {scores['mean']:.3f}")
```

## Next Steps

### 1. Customize Your Prompts

Create domain-specific prompts for your use case:

```python
pm.register_prompt(
    name="medical_qa_prompt",
    template="""You are a medical information assistant. Use the context to answer.

Context: {context}

Question: {question}

Important: Only use information from the context. For medical advice, recommend consulting a doctor.

Answer:""",
    description="Medical QA with safety guidelines",
    parameters=["context", "question"],
    tags={"domain": "medical", "safety": "high"}
)
```

### 2. Implement A/B Testing

Compare different prompt versions:

```python
comparison = pm.compare_prompts(
    prompt_names=["my_rag_prompt:v1", "my_rag_prompt:v2"],
    test_inputs=[{"context": "...", "question": "..."}],
    evaluation_fn=lambda p, i: judge.evaluate_faithfulness(...).score
)
```

### 3. Set Up Continuous Evaluation

Create a scheduled job to evaluate your production prompts:

```python
# In a Databricks job notebook
# Schedule: Daily at 1 AM

# Load production data
production_examples = spark.table("main.rag.production_examples").toPandas()

# Convert to RAGExample objects
examples = [
    RAGExample(
        question=row["question"],
        context=row["context"],
        answer=row["answer"]
    )
    for _, row in production_examples.iterrows()
]

# Evaluate
results = pipeline.evaluate_dataset(examples, log_to_mlflow=True)

# Alert if quality drops
if results["aggregate_scores"]["faithfulness"]["mean"] < 0.7:
    # Send alert (email, Slack, etc.)
    print("⚠️ Quality alert: Faithfulness below threshold!")
```

### 4. Integrate with Model Serving

Deploy your best prompt in a serving endpoint:

```python
# In your serving endpoint code
prompt = pm.get_prompt("my_rag_prompt", version="latest")

def predict(context: str, question: str) -> str:
    formatted_prompt = pm.format_prompt(
        prompt,
        context=context,
        question=question
    )
    # Call your LLM
    return llm.generate(formatted_prompt)
```

## Common Use Cases

### Use Case 1: Customer Support QA

```python
support_prompt = pm.register_prompt(
    name="customer_support_qa",
    template="""You are a helpful customer support assistant. Answer based on the knowledge base.

Knowledge Base: {context}

Customer Question: {question}

Provide a friendly, accurate answer. If unsure, offer to connect them with a human agent.

Answer:""",
    parameters=["context", "question"],
    tags={"department": "support", "tone": "friendly"}
)
```

### Use Case 2: Technical Documentation QA

```python
tech_prompt = pm.register_prompt(
    name="technical_docs_qa",
    template="""Answer the technical question using the documentation below.

Documentation: {context}

Question: {question}

Provide a precise, technical answer with code examples if applicable.

Answer:""",
    parameters=["context", "question"],
    tags={"domain": "technical", "format": "code-friendly"}
)
```

### Use Case 3: Research Summarization

```python
research_prompt = pm.register_prompt(
    name="research_summarization",
    template="""Summarize the research findings below to answer the question.

Research Papers: {context}

Question: {question}

Provide an evidence-based summary citing specific studies.

Summary:""",
    parameters=["context", "question"],
    tags={"domain": "research", "citation": "required"}
)
```

## Monitoring Dashboard

View your evaluation results in MLflow:

1. Go to your Databricks workspace
2. Navigate to "Machine Learning" → "Experiments"
3. Open your experiment (e.g., `/Users/your-email/rag_evaluation`)
4. View metrics, parameters, and artifacts for each run

## Troubleshooting

### Issue: "Module not found"
```python
# Ensure path is correct
import sys
sys.path.append("/Workspace/Repos/your-username/dolly")
```

### Issue: "Unity Catalog permission denied"
```sql
-- Grant necessary permissions
GRANT CREATE ON CATALOG main TO `your-user@company.com`;
GRANT USE ON SCHEMA main.prompts TO `your-user@company.com`;
```

### Issue: "Foundation Model API not available"
- Check your workspace has Foundation Model APIs enabled
- Verify your user has access to the model
- Contact your workspace admin if needed

## Best Practices

1. **Version Everything**: Always tag and version your prompts
2. **Test Before Deploy**: Evaluate on diverse examples before production
3. **Monitor Continuously**: Set up scheduled evaluation jobs
4. **Use Appropriate Models**: Llama 3 70B for quality, 8B for speed
5. **Set Thresholds**: Define minimum acceptable scores for each metric

## Resources

- **Full Example**: `/examples/prompt_management_evaluation_example.py`
- **Documentation**: `/genai_evaluation/README.md`
- **Configuration**: `/genai_evaluation/config.py`

## Support

Need help? Check:
- [Databricks GenAI Documentation](https://docs.databricks.com/generative-ai)
- [Databricks Community Forums](https://community.databricks.com)
- [GitHub Issues](https://github.com/databrickslabs/dolly/issues)

---

**Ready to build production GenAI applications on Databricks!** 🚀

"""
Example: RAG Prompt Management and Evaluation on Databricks

This example demonstrates how to:
1. Register and manage RAG prompts in Unity Catalog
2. Evaluate RAG outputs using LLM-as-judge
3. Compare different prompt versions
4. Track experiments with MLflow

Run this on a Databricks cluster with:
- DBR 14.3 LTS ML or higher
- Access to Foundation Model APIs
- Unity Catalog enabled
"""

# COMMAND ----------
# MAGIC %pip install mlflow>=2.9.0 databricks-sdk>=0.18.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import sys
sys.path.append("/Workspace/Repos/your-username/dolly")  # Adjust path as needed

from genai_evaluation.prompt_manager import PromptManager, PromptTemplate
from genai_evaluation.rag_evaluator import (
    LLMJudge,
    RAGEvaluationPipeline,
    RAGExample,
    RAGMetric
)
import mlflow

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Initialize Prompt Manager
# MAGIC
# MAGIC Configure Unity Catalog location and MLflow experiment for prompt management.

# COMMAND ----------
# Initialize Prompt Manager
prompt_manager = PromptManager(
    catalog="main",  # Your Unity Catalog catalog
    schema="prompts",  # Schema for storing prompts
    mlflow_experiment="/Users/your-email/rag_prompts"  # MLflow experiment path
)

print("Prompt Manager initialized")
print(f"Namespace: {prompt_manager.namespace}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Register RAG Prompt Templates
# MAGIC
# MAGIC Create and version multiple prompt templates for RAG question-answering.

# COMMAND ----------
# Register a basic RAG prompt
basic_prompt = prompt_manager.register_prompt(
    name="rag_qa_basic",
    template="""Use the following context to answer the question. If you cannot answer based on the context, say "I don't know."

Context: {context}

Question: {question}

Answer:""",
    description="Basic RAG QA prompt with context and question",
    parameters=["context", "question"],
    tags={
        "use_case": "qa",
        "version_type": "basic",
        "model": "llama3"
    },
    metadata={
        "author": "data_team",
        "tested": True,
        "avg_latency_ms": 450
    }
)

print(f"Registered prompt: {basic_prompt.name} v{basic_prompt.version}")

# COMMAND ----------
# Register an enhanced RAG prompt with instructions
enhanced_prompt = prompt_manager.register_prompt(
    name="rag_qa_enhanced",
    template="""You are a helpful AI assistant. Use the context below to provide an accurate, complete answer to the question.

Guidelines:
- Only use information from the provided context
- Be concise but thorough
- If the context doesn't contain enough information, clearly state what is missing
- Cite specific parts of the context when relevant

Context:
{context}

Question: {question}

Answer:""",
    description="Enhanced RAG prompt with detailed instructions",
    parameters=["context", "question"],
    tags={
        "use_case": "qa",
        "version_type": "enhanced",
        "model": "llama3"
    },
    metadata={
        "author": "data_team",
        "tested": False,
        "expected_improvement": "10-15% over basic"
    }
)

print(f"Registered prompt: {enhanced_prompt.name} v{enhanced_prompt.version}")

# COMMAND ----------
# Register a chain-of-thought RAG prompt
cot_prompt = prompt_manager.register_prompt(
    name="rag_qa_cot",
    template="""Use the following context to answer the question. Think step by step.

Context: {context}

Question: {question}

Let's approach this systematically:
1. First, identify the relevant information in the context
2. Then, reason about how it relates to the question
3. Finally, formulate a clear answer

Answer:""",
    description="Chain-of-thought RAG prompt for complex questions",
    parameters=["context", "question"],
    tags={
        "use_case": "qa",
        "version_type": "cot",
        "model": "llama3"
    },
    metadata={
        "author": "data_team",
        "tested": False,
        "best_for": "complex_reasoning"
    }
)

print(f"Registered prompt: {cot_prompt.name} v{cot_prompt.version}")

# COMMAND ----------
# List all registered prompts
all_prompts = prompt_manager.list_prompts()
print(f"\nTotal registered prompts: {len(all_prompts)}")

# Filter by tag
qa_prompts = prompt_manager.list_prompts(tag_filter={"use_case": "qa"})
print(f"QA prompts: {len(qa_prompts)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Initialize LLM Judge for Evaluation
# MAGIC
# MAGIC Set up the LLM-as-judge evaluator using Databricks Foundation Models.

# COMMAND ----------
# Initialize LLM Judge with Llama 3 70B
judge = LLMJudge(
    model_name="databricks-meta-llama-3-70b-instruct",
    # endpoint_name="your-serving-endpoint",  # Optional: use custom endpoint
    temperature=0.0,  # Deterministic for consistent evaluation
    max_tokens=500
)

print(f"LLM Judge initialized with model: {judge.model_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Create Test Examples
# MAGIC
# MAGIC Prepare RAG examples for evaluation with questions, context, and answers.

# COMMAND ----------
# Create test RAG examples
test_examples = [
    RAGExample(
        question="What is the capital of France?",
        context="Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023, Paris is the fourth-largest city in the European Union.",
        answer="The capital of France is Paris, which is also its most populous city.",
        metadata={"domain": "geography", "difficulty": "easy"}
    ),
    RAGExample(
        question="When was the Eiffel Tower built?",
        context="The Eiffel Tower was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. It was designed by Gustave Eiffel's company.",
        answer="The Eiffel Tower was built between 1887 and 1889.",
        metadata={"domain": "history", "difficulty": "easy"}
    ),
    RAGExample(
        question="What is machine learning?",
        context="Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being programmed to do so.",
        answer="Machine learning is a type of AI that allows applications to learn from data and improve accuracy automatically, without explicit programming for each task.",
        metadata={"domain": "technology", "difficulty": "medium"}
    ),
    RAGExample(
        question="How does photosynthesis work?",
        context="Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll.",
        answer="Photosynthesis is the process where plants use sunlight, carbon dioxide, and water to create nutrients. This process involves chlorophyll, the green pigment in plants.",
        metadata={"domain": "biology", "difficulty": "medium"}
    ),
    RAGExample(
        question="What caused World War I?",
        context="The assassination of Archduke Franz Ferdinand of Austria-Hungary was the immediate trigger of World War I. However, underlying causes included militarism, alliances, imperialism, and nationalism.",
        answer="World War I was triggered by the assassination of Archduke Franz Ferdinand, but the underlying causes were complex, including militarism, alliances, imperialism, and nationalism.",
        metadata={"domain": "history", "difficulty": "hard"}
    )
]

print(f"Created {len(test_examples)} test examples")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Evaluate Single Example
# MAGIC
# MAGIC Demonstrate evaluation of a single RAG response across multiple metrics.

# COMMAND ----------
# Evaluate a single example across all metrics
example = test_examples[0]
print(f"Question: {example.question}")
print(f"Answer: {example.answer}\n")

results = judge.evaluate_all(example)

print("Evaluation Results:")
print("-" * 60)
for metric, result in results.items():
    print(f"\n{metric.value.upper()}:")
    print(f"  Score: {result.score:.2f}")
    print(f"  Reasoning: {result.reasoning}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Evaluate Full Dataset
# MAGIC
# MAGIC Run evaluation pipeline across all test examples with parallel processing.

# COMMAND ----------
# Initialize evaluation pipeline
eval_pipeline = RAGEvaluationPipeline(
    judge=judge,
    mlflow_experiment="/Users/your-email/rag_evaluation",
    parallel_workers=4  # Adjust based on your cluster size
)

# Evaluate all examples
evaluation_results = eval_pipeline.evaluate_dataset(
    examples=test_examples,
    metrics=[
        RAGMetric.FAITHFULNESS,
        RAGMetric.ANSWER_RELEVANCE,
        RAGMetric.CONTEXT_RELEVANCE,
        RAGMetric.GROUNDEDNESS
    ],
    log_to_mlflow=True
)

# Display aggregate scores
print("\nAggregate Evaluation Scores:")
print("=" * 60)
for metric, scores in evaluation_results["aggregate_scores"].items():
    print(f"\n{metric.upper()}:")
    print(f"  Mean:  {scores['mean']:.3f}")
    print(f"  Min:   {scores['min']:.3f}")
    print(f"  Max:   {scores['max']:.3f}")
    print(f"  Count: {scores['count']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Compare Prompt Versions
# MAGIC
# MAGIC A/B test different prompt templates on the same questions.

# COMMAND ----------
# Simulate responses from different prompt versions
# In production, you would actually call your RAG system with each prompt

system_a_examples = [
    RAGExample(
        question=ex.question,
        context=ex.context,
        answer=ex.answer,  # Response using basic prompt
        metadata={**ex.metadata, "prompt_version": "basic"}
    )
    for ex in test_examples
]

system_b_examples = [
    RAGExample(
        question=ex.question,
        context=ex.context,
        answer=ex.answer + " This response uses more detail.",  # Simulate enhanced
        metadata={**ex.metadata, "prompt_version": "enhanced"}
    )
    for ex in test_examples
]

# Compare systems
comparison_results = eval_pipeline.compare_systems(
    system_results={
        "basic_prompt": system_a_examples,
        "enhanced_prompt": system_b_examples
    },
    metrics=[RAGMetric.FAITHFULNESS, RAGMetric.ANSWER_RELEVANCE]
)

# Display comparison
print("\nPrompt Comparison Results:")
print("=" * 60)
for system_name, metrics in comparison_results.items():
    print(f"\n{system_name.upper()}:")
    for metric_name, scores in metrics.items():
        print(f"  {metric_name}: {scores['mean']:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Custom Evaluation Function
# MAGIC
# MAGIC Create a custom evaluator for specific metrics (e.g., token efficiency).

# COMMAND ----------
def token_efficiency_score(prompt_template, test_input):
    """
    Custom metric: Evaluate prompt token efficiency.
    Lower token count while maintaining quality = higher score.
    """
    formatted = prompt_manager.format_prompt(prompt_template, **test_input)
    token_count = len(formatted.split())  # Simplified token counting

    # Score inversely proportional to token count
    # Normalize to 0-1 range (assuming max 1000 tokens)
    efficiency = max(0, 1 - (token_count / 1000))

    return efficiency

# Compare prompts on token efficiency
test_input = {
    "context": "Sample context text here...",
    "question": "What is the main topic?"
}

prompt_comparison = prompt_manager.compare_prompts(
    prompt_names=["rag_qa_basic", "rag_qa_enhanced", "rag_qa_cot"],
    test_inputs=[test_input] * 3,
    evaluation_fn=token_efficiency_score
)

print("\nToken Efficiency Comparison:")
print("=" * 60)
for prompt_name, results in prompt_comparison.items():
    print(f"{prompt_name}: {results['avg_score']:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Monitor and Iterate
# MAGIC
# MAGIC Track prompt performance over time and iterate based on evaluation results.

# COMMAND ----------
# Retrieve best performing prompt
best_prompt_name = max(
    evaluation_results["aggregate_scores"].items(),
    key=lambda x: x[1]["mean"]
)[0]

print(f"\nBest performing metric: {best_prompt_name}")
print(f"Score: {evaluation_results['aggregate_scores'][best_prompt_name]['mean']:.3f}")

# Register new version based on learnings
improved_prompt = prompt_manager.register_prompt(
    name="rag_qa_production",
    template="""[Your improved prompt template based on evaluation results]

Context: {context}

Question: {question}

Answer:""",
    description="Production RAG prompt - v1 based on evaluation results",
    parameters=["context", "question"],
    tags={
        "use_case": "qa",
        "version_type": "production",
        "status": "active",
        "model": "llama3"
    },
    metadata={
        "based_on": "rag_qa_enhanced",
        "improvement_over_basic": "15%",
        "evaluation_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None
    }
)

print(f"\nRegistered production prompt: {improved_prompt.name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Best Practices Summary
# MAGIC
# MAGIC Key takeaways for prompt management and evaluation:
# MAGIC
# MAGIC ### Prompt Management
# MAGIC - Version all prompts with descriptive tags
# MAGIC - Track metadata: author, performance metrics, use cases
# MAGIC - Use semantic versioning for major changes
# MAGIC - Document prompt requirements and parameters
# MAGIC
# MAGIC ### Evaluation
# MAGIC - Evaluate on diverse, representative examples
# MAGIC - Use multiple metrics (faithfulness, relevance, etc.)
# MAGIC - Run evaluations in parallel for efficiency
# MAGIC - Track all experiments in MLflow
# MAGIC - Compare prompt versions with statistical significance
# MAGIC
# MAGIC ### Production Deployment
# MAGIC - A/B test prompt changes before full rollout
# MAGIC - Monitor prompt performance continuously
# MAGIC - Set up alerts for quality degradation
# MAGIC - Maintain a prompt registry with governance
# MAGIC - Document evaluation criteria and thresholds

# COMMAND ----------
print("Example notebook completed successfully!")
print("\nNext steps:")
print("1. Customize prompts for your specific use case")
print("2. Create domain-specific evaluation examples")
print("3. Set up automated evaluation pipelines")
print("4. Integrate with your RAG production system")
print("5. Configure monitoring and alerting")

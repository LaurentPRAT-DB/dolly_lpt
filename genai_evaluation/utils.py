"""
Utility functions for GenAI Evaluation Framework

Common helper functions for prompt formatting, data processing, and result analysis.
"""

from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime


def clean_json_response(response: str) -> str:
    """
    Clean LLM response to extract valid JSON.

    Handles common cases like markdown code blocks and extra text.

    Args:
        response: Raw LLM response

    Returns:
        Cleaned JSON string
    """
    response = response.strip()

    # Remove markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]

    # Try to extract JSON object using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response)
    if matches:
        response = matches[0]

    return response.strip()


def estimate_tokens(text: str, method: str = "simple") -> int:
    """
    Estimate token count for text.

    Args:
        text: Input text
        method: Estimation method ("simple", "words", "chars")

    Returns:
        Estimated token count
    """
    if method == "simple":
        # Rough approximation: 1 token ~ 4 characters
        return len(text) // 4
    elif method == "words":
        # Approximation: 1 token ~ 0.75 words
        return int(len(text.split()) * 1.33)
    elif method == "chars":
        # Character-based with punctuation consideration
        return len(text.split()) + text.count(",") + text.count(".")
    else:
        return len(text.split())


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separator: str = "\n"
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in tokens
        overlap: Number of overlapping tokens between chunks
        separator: Preferred split separator

    Returns:
        List of text chunks
    """
    # Split by separator
    segments = text.split(separator)

    chunks = []
    current_chunk = []
    current_size = 0

    for segment in segments:
        segment_size = estimate_tokens(segment)

        if current_size + segment_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(separator.join(current_chunk))

            # Start new chunk with overlap
            overlap_segments = []
            overlap_size = 0
            for seg in reversed(current_chunk):
                seg_size = estimate_tokens(seg)
                if overlap_size + seg_size <= overlap:
                    overlap_segments.insert(0, seg)
                    overlap_size += seg_size
                else:
                    break

            current_chunk = overlap_segments
            current_size = overlap_size

        current_chunk.append(segment)
        current_size += segment_size

    # Add remaining chunk
    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def calculate_aggregate_metrics(
    results: List[Dict[str, Any]],
    metric_keys: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate aggregate statistics for metrics.

    Args:
        results: List of result dictionaries
        metric_keys: Keys to aggregate

    Returns:
        Dictionary with aggregate statistics
    """
    aggregates = {}

    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregates[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "median": sorted(values)[len(values) // 2],
                "count": len(values),
                "std": (
                    sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)
                ) ** 0.5
            }

    return aggregates


def format_evaluation_report(
    aggregate_scores: Dict[str, Dict[str, float]],
    detailed_results: Optional[List[Dict]] = None,
    include_examples: bool = True,
    max_examples: int = 5
) -> str:
    """
    Format evaluation results into a readable report.

    Args:
        aggregate_scores: Aggregate metric scores
        detailed_results: Optional detailed per-example results
        include_examples: Whether to include example details
        max_examples: Maximum number of examples to include

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("RAG EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Aggregate scores
    report.append("AGGREGATE SCORES")
    report.append("-" * 80)
    for metric, scores in aggregate_scores.items():
        report.append(f"\n{metric.upper()}:")
        report.append(f"  Mean:   {scores['mean']:.3f}")
        report.append(f"  Median: {scores.get('median', 'N/A')}")
        report.append(f"  Min:    {scores['min']:.3f}")
        report.append(f"  Max:    {scores['max']:.3f}")
        report.append(f"  Std:    {scores.get('std', 'N/A')}")
        report.append(f"  Count:  {scores['count']}")

    # Detailed examples
    if include_examples and detailed_results:
        report.append("\n")
        report.append("EXAMPLE EVALUATIONS")
        report.append("-" * 80)

        for i, result in enumerate(detailed_results[:max_examples]):
            report.append(f"\nExample {i+1}:")
            if "question" in result:
                report.append(f"  Question: {result['question'][:100]}...")
            if "scores" in result:
                for metric, score_info in result["scores"].items():
                    report.append(f"  {metric}: {score_info['score']:.3f}")
                    report.append(f"    Reasoning: {score_info['reasoning'][:150]}...")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def create_prompt_template(
    system_message: str,
    user_template: str,
    parameters: List[str],
    few_shot_examples: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Create a structured prompt template with system message and examples.

    Args:
        system_message: System instruction
        user_template: User message template with {parameter} placeholders
        parameters: List of required parameters
        few_shot_examples: Optional list of example input-output pairs

    Returns:
        Structured prompt template dictionary
    """
    template = {
        "system": system_message,
        "user_template": user_template,
        "parameters": parameters,
        "few_shot_examples": few_shot_examples or [],
        "created_at": datetime.now().isoformat()
    }

    return template


def validate_rag_example(example: Dict[str, Any]) -> bool:
    """
    Validate that a RAG example has required fields.

    Args:
        example: Example dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["question", "context", "answer"]
    return all(field in example for field in required_fields)


def batch_examples(
    examples: List[Any],
    batch_size: int
) -> List[List[Any]]:
    """
    Split examples into batches.

    Args:
        examples: List of examples
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [
        examples[i:i + batch_size]
        for i in range(0, len(examples), batch_size)
    ]


def merge_contexts(
    contexts: List[str],
    max_length: int = 4096,
    separator: str = "\n\n"
) -> str:
    """
    Merge multiple context passages, respecting max length.

    Args:
        contexts: List of context passages
        max_length: Maximum total length in characters
        separator: Separator between contexts

    Returns:
        Merged context string
    """
    merged = []
    current_length = 0

    for ctx in contexts:
        ctx_length = len(ctx) + len(separator)
        if current_length + ctx_length > max_length:
            break
        merged.append(ctx)
        current_length += ctx_length

    return separator.join(merged)


def extract_citations(
    answer: str,
    context: str
) -> List[str]:
    """
    Extract parts of the answer that can be cited from context.

    Args:
        answer: Generated answer
        context: Source context

    Returns:
        List of cited phrases
    """
    citations = []

    # Split answer into sentences
    sentences = re.split(r'[.!?]+', answer)

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.lower() in context.lower():
            citations.append(sentence)

    return citations


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    Calculate simple similarity score between two texts.

    Uses word overlap as a basic similarity metric.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

# bleu_evaluation.py
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def calculate_bleu_scores(generated_summaries_file: str, reference_summaries_file: str = None):
    """
    Calculate BLEU scores for generated summaries.
    
    Args:
        generated_summaries_file: Path to file with AI-generated summaries
        reference_summaries_file: Path to file with reference summaries (if available)
    """
    # Load generated summaries
    with open(generated_summaries_file, 'r') as f:
        generated_data = json.load(f)
    
    # If you have reference summaries, load them for comparison
    # For now, we'll calculate self-BLEU scores between different sentiment groups
    
    sentiment_texts = []
    for sentiment, summary in generated_data.items():
        if summary and summary != "No comments found for this sentiment category.":
            sentiment_texts.append(word_tokenize(summary.lower()))
    
    # Calculate pairwise BLEU scores
    bleu_scores = []
    for i, text1 in enumerate(sentiment_texts):
        for j, text2 in enumerate(sentiment_texts):
            if i != j:
                score = sentence_bleu([text1], text2)
                bleu_scores.append(score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"Average BLEU score between sentiment summaries: {avg_bleu:.4f}")
    
    return avg_bleu

# Usage
# calculate_bleu_scores("analysis_results/sentiment_summaries.json")

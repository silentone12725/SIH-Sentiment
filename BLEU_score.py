# enhanced_bleu_evaluation.py
import pandas as pd
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict

# Download required NLTK data - UPDATED FOR NEWER NLTK VERSIONS
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)  # Fallback for older versions
except:
    pass

class BLEUEvaluator:
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def load_summaries(self, file_path):
        """Load generated summaries from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        if not text or text == "No comments found for this sentiment category.":
            return []
        
        # Clean text and tokenize
        text = text.lower()
        text = text.replace('\n', ' ')
        
        # Try different tokenization methods for compatibility
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback tokenization if punkt_tab is not available
            print("⚠️ Using fallback tokenization method")
            tokens = text.split()
        
        # Remove very short tokens and punctuation
        tokens = [token for token in tokens if len(token) > 1 and token.isalnum()]
        return tokens
    
    def calculate_sentiment_bleu_scores(self, summaries_dict):
        """Calculate BLEU scores between different sentiment summaries"""
        print("🔄 Calculating BLEU Scores Between Sentiment Groups...")
        print("=" * 60)
        
        # Preprocess all summaries
        processed_summaries = {}
        for sentiment, summary in summaries_dict.items():
            tokens = self.preprocess_text(summary)
            if tokens:  # Only include non-empty summaries
                processed_summaries[sentiment] = tokens
        
        if len(processed_summaries) < 2:
            print("❌ Need at least 2 non-empty summaries for comparison")
            return {}
        
        # Calculate pairwise BLEU scores
        results = {}
        sentiment_pairs = list(processed_summaries.keys())
        
        for i, sentiment1 in enumerate(sentiment_pairs):
            for j, sentiment2 in enumerate(sentiment_pairs):
                if i != j:  # Don't compare with itself
                    reference = processed_summaries[sentiment1]
                    candidate = processed_summaries[sentiment2]
                    
                    # Calculate BLEU score with smoothing for better results
                    bleu_score = sentence_bleu(
                        [reference], 
                        candidate, 
                        smoothing_function=self.smoothing.method1
                    )
                    
                    pair_name = f"{sentiment1}_vs_{sentiment2}"
                    results[pair_name] = bleu_score
                    
                    print(f"📊 {sentiment1} vs {sentiment2}: {bleu_score:.4f}")
        
        return results
    
    def calculate_corpus_bleu(self, summaries_dict):
        """Calculate corpus-level BLEU score"""
        processed_summaries = []
        
        for sentiment, summary in summaries_dict.items():
            tokens = self.preprocess_text(summary)
            if tokens:
                processed_summaries.append(tokens)
        
        if len(processed_summaries) < 2:
            return 0.0
        
        # Use first summary as reference, others as candidates
        references = [processed_summaries[0]]
        candidates = processed_summaries[1:]
        
        corpus_bleu_score = corpus_bleu(
            [references] * len(candidates),
            candidates,
            smoothing_function=self.smoothing.method1
        )
        
        return corpus_bleu_score
    
    def analyze_summary_diversity(self, summaries_dict):
        """Analyze diversity between summaries"""
        print("\n🎯 Summary Diversity Analysis:")
        print("=" * 40)
        
        bleu_scores = self.calculate_sentiment_bleu_scores(summaries_dict)
        
        if not bleu_scores:
            return
        
        # Calculate statistics
        scores_list = list(bleu_scores.values())
        avg_score = np.mean(scores_list)
        min_score = np.min(scores_list)
        max_score = np.max(scores_list)
        std_score = np.std(scores_list)
        
        print(f"\n📈 BLEU Score Statistics:")
        print(f"   Average: {avg_score:.4f}")
        print(f"   Minimum: {min_score:.4f}")
        print(f"   Maximum: {max_score:.4f}")
        print(f"   Std Dev: {std_score:.4f}")
        
        # Interpret results
        print(f"\n💡 Interpretation:")
        if avg_score < 0.1:
            print("   🎊 High Diversity: Summaries are quite different (Good!)")
        elif avg_score < 0.3:
            print("   📊 Moderate Diversity: Some similarities between summaries")
        else:
            print("   ⚠️  Low Diversity: Summaries are quite similar")
        
        return {
            'average_bleu': avg_score,
            'min_bleu': min_score,
            'max_bleu': max_score,
            'std_bleu': std_score,
            'pairwise_scores': bleu_scores
        }

def main():
    """Main function to calculate BLEU scores for your analysis"""
    print("🚀 BLEU Score Calculator for MCA Analysis")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = BLEUEvaluator()
    
    # Load your generated summaries
    summaries_file = "sentiment_summaries.json"  # Updated path
    summaries = evaluator.load_summaries(summaries_file)
    
    if not summaries:
        print("❌ Could not load summaries file")
        return
    
    print(f"✅ Loaded summaries for {len(summaries)} sentiment groups")
    
    # Calculate BLEU scores between sentiment groups
    diversity_results = evaluator.analyze_summary_diversity(summaries)
    
    # Calculate corpus BLEU
    corpus_bleu = evaluator.calculate_corpus_bleu(summaries)
    print(f"\n📊 Corpus BLEU Score: {corpus_bleu:.4f}")
    
    # Save results
    if diversity_results:
        results_file = "bleu_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'diversity_analysis': diversity_results,
                'corpus_bleu': corpus_bleu,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    print("\n✨ BLEU evaluation complete!")

if __name__ == "__main__":
    main()

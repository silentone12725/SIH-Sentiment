import pandas as pd
import requests
import json
import time
import re
from typing import List, Dict, Any
from datetime import datetime
import os

# Add these imports for Hugging Face support
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")


class StakeholderCommentAnalyzer:
    def __init__(self, 
                 model_type: str = "ollama",  # "ollama" or "huggingface"
                 ollama_url: str = "http://localhost:11434", 
                 model: str = "llama3.1",
                 hf_model: str = "microsoft/DialoGPT-medium",
                 use_gpu: bool = True):
        """
        Initialize the analyzer with flexible model configuration.
        """
        self.model_type = model_type.lower()
        self.ollama_url = ollama_url
        self.model = model
        self.hf_model = hf_model
        self.use_gpu = use_gpu

        # Initialize model based on type
        if self.model_type == "ollama":
            self.api_endpoint = f"{ollama_url}/api/generate"
            self.hf_pipeline = None
        elif self.model_type == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("Transformers library not available. Install with: pip install transformers torch")
            self._initialize_hf_model()
            self.api_endpoint = None
        else:
            raise ValueError("model_type must be either 'ollama' or 'huggingface'")

    def _initialize_hf_model(self):
        """Initialize Hugging Face model pipeline"""
        try:
            print(f"Loading Hugging Face model: {self.hf_model}")
            device = 0 if self.use_gpu else -1

            if "gpt" in self.hf_model.lower() or "llama" in self.hf_model.lower():
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    device=device,
                    torch_dtype="auto",
                    trust_remote_code=True
                )
            else:
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    device=device
                )
            print("Hugging Face model loaded successfully!")
        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            print("Falling back to distilgpt2...")
            self.hf_pipeline = pipeline("text-generation", model="distilgpt2", device=device)

    def call_model_api(self, prompt: str, system_prompt: str = None) -> str:
        """Universal method to call either Ollama or Hugging Face model."""
        if self.model_type == "ollama":
            return self._call_ollama_api(prompt, system_prompt)
        elif self.model_type == "huggingface":
            return self._call_hf_model(prompt, system_prompt)

    def _call_ollama_api(self, prompt: str, system_prompt: str = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 2000
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            print("Calling Ollama API for analysis...")
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: Unable to generate summary due to API error: {e}"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Error: Unexpected error occurred: {e}"

    def _call_hf_model(self, prompt: str, system_prompt: str = None) -> str:
        try:
            print("Calling Hugging Face model for analysis...")
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            if len(full_prompt) > 2000:
                full_prompt = full_prompt[:2000] + "..."

            response = self.hf_pipeline(
                full_prompt,
                max_length=len(full_prompt) + 500,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.hf_pipeline.tokenizer.eos_token_id
            )

            generated_text = response[0]['generated_text']
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()
            return generated_text
        except Exception as e:
            print(f"Error calling Hugging Face model: {e}")
            return f"Error: Unable to generate summary due to HF model error: {e}"

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(df)} comments")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def clean_comment(self, comment: str) -> str:
        if pd.isna(comment):
            return ""
        comment = re.sub(r'\s+', ' ', str(comment))
        comment = comment.replace('\n', ' ').replace('\r', ' ')
        comment = re.sub(r'\s*\.\s*', '. ', comment)
        comment = re.sub(r'\s*,\s*', ', ', comment)
        comment = re.sub(r'\([^)]*\)', '', comment)
        return comment.strip()

    def group_comments_by_sentiment(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        sentiment_groups = {
            'Positive': [],
            'Negative': [],
            'Neutral': [],
            'Mixed': []
        }
        for _, row in df.iterrows():
            cleaned_comment = self.clean_comment(row['comment'])
            if cleaned_comment:
                sentiment = row['expected_sentiment']
                if sentiment in sentiment_groups:
                    sentiment_groups[sentiment].append({
                        'comment': cleaned_comment,
                        'stakeholder': row['stakeholder_name'],
                        'draft_id': row['draft_id'],
                        'section': row['draft_section']
                    })
        return sentiment_groups

    def create_system_prompt(self) -> str:
        return """You are an expert legal and policy analyst specializing in corporate law and stakeholder consultation analysis...
        Provide balanced, objective analysis that would help policymakers understand stakeholder perspectives."""

    def generate_prompt_for_group(self, sentiment: str, comments: List[Dict], sample_size: int = 50) -> str:
        sample_comments = comments[:sample_size] if len(comments) > sample_size else comments
        comments_text = "\n\n".join([
            f"Comment {i+1} (Draft {comment['draft_id']}, {comment['section']}):\n{comment['comment']}"
            for i, comment in enumerate(sample_comments)
        ])
        prompt = f"""
        Analyze the following {sentiment.lower()} stakeholder comments about proposed Companies Act amendments. There are {len(comments)} total comments in this sentiment category, showing {len(sample_comments)} samples.
        {comments_text}
        Provide a structured summary including key themes, patterns, section-specific issues, stakeholder priorities, recommendations, business impact, implementation concerns, and overall sentiment.
        """
        return prompt

    def analyze_all_sentiment_groups(self, df: pd.DataFrame) -> Dict[str, str]:
        sentiment_groups = self.group_comments_by_sentiment(df)
        system_prompt = self.create_system_prompt()
        summaries = {}
        for sentiment, comments in sentiment_groups.items():
            if not comments:
                summaries[sentiment] = "No comments found for this sentiment category."
                continue
            print(f"\nAnalyzing {sentiment} sentiment group ({len(comments)} comments)...")
            prompt = self.generate_prompt_for_group(sentiment, comments)
            summary = self.call_model_api(prompt, system_prompt)
            summaries[sentiment] = summary
            time.sleep(2)
        return summaries

    def generate_overall_summary(self, df: pd.DataFrame, sentiment_summaries: Dict[str, str]) -> str:
        total_comments = len(df)
        draft_counts = df['draft_id'].value_counts().to_dict()
        section_counts = df['draft_section'].value_counts().to_dict()
        sentiment_counts = df['expected_sentiment'].value_counts().to_dict()
        overall_prompt = f"""
        Based on analysis of {total_comments} comments, synthesize insights across sentiment categories.
        Draft Distribution: {draft_counts}
        Section Distribution: {section_counts}
        Sentiment Distribution: {sentiment_counts}
        POSITIVE: {sentiment_summaries.get('Positive','N/A')}
        NEGATIVE: {sentiment_summaries.get('Negative','N/A')}
        NEUTRAL: {sentiment_summaries.get('Neutral','N/A')}
        MIXED: {sentiment_summaries.get('Mixed','N/A')}
        Provide an executive summary highlighting critical issues, consensus, contentious areas, MSME impact, recommendations, and policy implications.
        """
        return self.call_model_api(overall_prompt, self.create_system_prompt())

    def save_results(self, df: pd.DataFrame, sentiment_summaries: Dict[str, str], overall_summary: str, output_dir: str = "analysis_results") -> None:
        os.makedirs(output_dir, exist_ok=True)
        enhanced_df = df.copy()
        enhanced_df['analysis_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sentiment_mapping = {k: v for k, v in sentiment_summaries.items()}
        enhanced_df['sentiment_group_summary'] = enhanced_df['expected_sentiment'].map(sentiment_mapping)
        enhanced_df.to_csv(os.path.join(output_dir, "enhanced_dataset_with_summaries.csv"), index=False)
        with open(os.path.join(output_dir, "sentiment_summaries.json"), 'w') as f:
            json.dump(sentiment_summaries, f, indent=2)
        with open(os.path.join(output_dir, "overall_summary.txt"), 'w') as f:
            f.write(f"MCA eConsultation Analysis - Overall Summary\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n\n")
            f.write(overall_summary)
        metadata = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_comments': len(df),
            'model_used': self.model,
            'sentiment_distribution': df['expected_sentiment'].value_counts().to_dict(),
            'draft_distribution': df['draft_id'].value_counts().to_dict(),
            'section_distribution': df['draft_section'].value_counts().to_dict()
        }
        with open(os.path.join(output_dir, "analysis_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Analysis results saved in: {output_dir}")


def main():
    MODEL_TYPE = "ollama"
    DATASET_PATH = "mca_consultation_test_data_max_realism_artifacts.csv"
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1"
    HF_MODEL = "microsoft/DialoGPT-medium"

    analyzer = StakeholderCommentAnalyzer(
        model_type=MODEL_TYPE,
        ollama_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        hf_model=HF_MODEL,
        use_gpu=True
    )

    print("Loading dataset...")
    df = analyzer.load_dataset(DATASET_PATH)
    if df is None:
        print("Failed to load dataset. Exiting.")
        return

    print(f"Dataset overview: {len(df)} comments, {df['stakeholder_name'].nunique()} stakeholders")
    sentiment_summaries = analyzer.analyze_all_sentiment_groups(df)
    overall_summary = analyzer.generate_overall_summary(df, sentiment_summaries)
    analyzer.save_results(df, sentiment_summaries, overall_summary)

    print("Analysis complete. Check 'analysis_results' directory for outputs.")


if __name__ == "__main__":
    main()

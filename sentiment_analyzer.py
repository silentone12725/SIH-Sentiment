import pandas as pd
import requests
import json
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os

# Add Hugging Face imports
try:
    from transformers import pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)

class OllamaSentimentAnalyzer:
    def __init__(self,
                 model_type: str = "ollama",
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.1",
                 hf_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the Sentiment Analyzer with support for multiple model types
        
        Args:
            model_type: "ollama" or "huggingface"
            base_url: Ollama API base URL
            model: Ollama model name
            hf_model: Hugging Face model name for sentiment analysis
        """
        self.model_type = model_type.lower()
        self.base_url = base_url
        self.model = model
        self.hf_model = hf_model
        self.api_url = f"{base_url}/api/generate"
        
        # Initialize model based on type
        if self.model_type == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers torch")
            self._initialize_hf_sentiment_model()
        
        # System prompt for sentiment analysis
        self.system_prompt = """You are an expert sentiment analysis AI specialized in analyzing stakeholder feedback on legal and regulatory documents.

Your task is to analyze comments submitted during public consultations on draft legislation and determine their sentiment.

Classification Categories:
- Positive: Comments that express approval, appreciation, support, or praise for provisions
- Negative: Comments that express criticism, concerns, objections, or identify problems
- Neutral: Comments that are informational, balanced, or provide neutral observations
- Mixed: Comments that contain both positive and negative sentiments

Guidelines:
1. Focus on the overall sentiment rather than individual phrases
2. Consider the context of legal/regulatory feedback
3. Pay attention to phrases like "highly appreciated", "welcome step", "too harsh", "impractical", "progressive approach"
4. For comments with multiple sentiments, determine if one predominates or if they're truly balanced (Mixed)
5. Respond with only one word: Positive, Negative, Neutral, or Mixed

Analyze the following stakeholder comment:"""

    def _initialize_hf_sentiment_model(self):
        """Initialize Hugging Face sentiment analysis model"""
        try:
            logging.info(f"Loading Hugging Face sentiment model: {self.hf_model}")
            self.hf_pipeline = pipeline(
                "sentiment-analysis",
                model=self.hf_model,
                device=0 if torch.cuda.is_available() else -1
            )
            logging.info("Hugging Face sentiment model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading HF model, using default: {e}")
            # Fallback to a basic sentiment model
            self.hf_pipeline = pipeline("sentiment-analysis")

    def test_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                logging.info("Successfully connected to Ollama API")
                return True
            else:
                logging.error(f"Failed to connect to Ollama API. Status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Connection error: {e}")
            return False

    def analyze_sentiment(self, comment: str) -> Optional[str]:
        """
        Analyze sentiment using either Ollama or Hugging Face
        
        Args:
            comment: The comment text to analyze
            
        Returns:
            Predicted sentiment or None if analysis fails
        """
        if self.model_type == "ollama":
            return self._analyze_sentiment_ollama(comment)
        elif self.model_type == "huggingface":
            return self._analyze_sentiment_hf(comment)

    def _analyze_sentiment_ollama(self, comment: str) -> Optional[str]:
        """Original Ollama sentiment analysis"""
        try:
            full_prompt = f"{self.system_prompt}\n\nComment: {comment}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 10,
                }
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                predicted_sentiment = result.get('response', '').strip()
                predicted_sentiment = self._clean_sentiment_response(predicted_sentiment)
                
                if predicted_sentiment in ['Positive', 'Negative', 'Neutral', 'Mixed']:
                    return predicted_sentiment
                else:
                    logging.warning(f"Invalid sentiment response: {predicted_sentiment}")
                    return None
            else:
                logging.error(f"API request failed with status code: {response.status_code}")
                return None

        except Exception as e:
            logging.error(f"Error in Ollama sentiment analysis: {e}")
            return None

    def _analyze_sentiment_hf(self, comment: str) -> Optional[str]:
        """Hugging Face sentiment analysis"""
        try:
            # Truncate comment if too long
            if len(comment) > 512:
                comment = comment[:512]
            
            result = self.hf_pipeline(comment)
            
            # Convert HF sentiment to our categories
            if isinstance(result, list):
                result = result[0]
            
            label = result['label'].upper()
            score = result['score']
            
            # Map common HF sentiment labels to our categories
            if label in ['POSITIVE', 'POS']:
                return 'Positive'
            elif label in ['NEGATIVE', 'NEG']:
                return 'Negative'
            elif label in ['NEUTRAL', 'NEU']:
                return 'Neutral'
            else:
                # For other models, use score to determine sentiment
                if score > 0.7:
                    return 'Positive' if 'POSITIVE' in label else 'Negative'
                else:
                    return 'Neutral'
                    
        except Exception as e:
            logging.error(f"Error in HF sentiment analysis: {e}")
            return None

    def _clean_sentiment_response(self, response: str) -> str:
        """Clean and extract sentiment from model response"""
        response = response.strip().lower()
        
        if 'positive' in response:
            return 'Positive'
        elif 'negative' in response:
            return 'Negative'
        elif 'neutral' in response:
            return 'Neutral'
        elif 'mixed' in response:
            return 'Mixed'
        else:
            first_word = response.split()[0] if response.split() else ""
            if first_word.capitalize() in ['Positive', 'Negative', 'Neutral', 'Mixed']:
                return first_word.capitalize()
        return response.capitalize()

    def process_dataset(self, df: pd.DataFrame, batch_size: int = 10,
                       delay_between_requests: float = 1.0) -> pd.DataFrame:
        """
        Process the entire dataset for sentiment analysis
        
        Args:
            df: DataFrame containing the comments
            batch_size: Number of comments to process before saving checkpoint
            delay_between_requests: Delay in seconds between API calls
            
        Returns:
            DataFrame with predicted sentiments added
        """
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Add column for predicted sentiment if it doesn't exist
        if 'predicted_sentiment' not in result_df.columns:
            result_df['predicted_sentiment'] = None
        
        # Track progress
        total_comments = len(result_df)
        processed_count = 0
        failed_count = 0
        
        logging.info(f"Starting sentiment analysis for {total_comments} comments")
        
        for index, row in result_df.iterrows():
            try:
                # Skip if already processed
                if pd.notna(result_df.at[index, 'predicted_sentiment']):
                    processed_count += 1
                    continue
                
                comment = row['comment']
                comment_id = row['comment_id']
                
                logging.info(f"Processing comment {processed_count + 1}/{total_comments} - ID: {comment_id}")
                
                # Analyze sentiment
                predicted_sentiment = self.analyze_sentiment(comment)
                
                if predicted_sentiment:
                    result_df.at[index, 'predicted_sentiment'] = predicted_sentiment
                    logging.info(f"Comment {comment_id}: {predicted_sentiment}")
                else:
                    result_df.at[index, 'predicted_sentiment'] = 'Error'
                    failed_count += 1
                    logging.warning(f"Failed to analyze comment {comment_id}")
                
                processed_count += 1
                
                # Save checkpoint every batch_size comments
                if processed_count % batch_size == 0:
                    self._save_checkpoint(result_df, processed_count)
                
                # Delay between requests to avoid overwhelming the API
                time.sleep(delay_between_requests)
                
            except KeyboardInterrupt:
                logging.info("Process interrupted by user. Saving current progress...")
                self._save_checkpoint(result_df, processed_count)
                break
            except Exception as e:
                logging.error(f"Error processing comment {row.get('comment_id', index)}: {e}")
                result_df.at[index, 'predicted_sentiment'] = 'Error'
                failed_count += 1
                processed_count += 1
        
        logging.info(f"Analysis complete. Processed: {processed_count}, Failed: {failed_count}")
        return result_df

    def _save_checkpoint(self, df: pd.DataFrame, processed_count: int):
        """Save a checkpoint of the current progress"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"sentiment_analysis_checkpoint_{processed_count}_{timestamp}.csv"
        df.to_csv(checkpoint_filename, index=False)
        logging.info(f"Checkpoint saved: {checkpoint_filename}")

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def save_results(df: pd.DataFrame, output_path: str):
    """Save the results to a CSV file"""
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def generate_analysis_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary report of the sentiment analysis"""
    report = {}
    
    # Overall sentiment distribution
    sentiment_counts = df['predicted_sentiment'].value_counts()
    report['sentiment_distribution'] = sentiment_counts.to_dict()
    
    # Sentiment distribution by draft
    draft_sentiment = df.groupby('draft_id')['predicted_sentiment'].value_counts().unstack(fill_value=0)
    report['sentiment_by_draft'] = draft_sentiment.to_dict()
    
    # Sentiment distribution by section
    section_sentiment = df.groupby('draft_section')['predicted_sentiment'].value_counts().unstack(fill_value=0)
    report['sentiment_by_section'] = section_sentiment.to_dict()
    
    # Errors and success rate
    total_comments = len(df)
    error_count = len(df[df['predicted_sentiment'] == 'Error'])
    success_rate = (total_comments - error_count) / total_comments * 100
    
    report['total_comments'] = total_comments
    report['successful_predictions'] = total_comments - error_count
    report['failed_predictions'] = error_count
    report['success_rate'] = success_rate
    
    return report

def main():
    """Main execution function"""
    
    # Configuration - Choose your model type
    MODEL_TYPE = "huggingface"  # Change to "ollama" for Ollama
    INPUT_FILE = "mca_consultation_test_data_max_realism_artifacts.csv"
    OUTPUT_FILE = f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    REPORT_FILE = f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Model configurations
    OLLAMA_URL = "http://localhost:11434" 
    OLLAMA_MODEL = "llama3.1"
    HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Good for sentiment analysis
    
    # Alternative HF models for sentiment:
    # - "nlptown/bert-base-multilingual-uncased-sentiment"
    # - "distilbert-base-uncased-finetuned-sst-2-english"
    # - "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Initialize analyzer
    analyzer = OllamaSentimentAnalyzer(
        model_type=MODEL_TYPE,
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        hf_model=HF_MODEL
    )

    # Test connection (only for Ollama)
    if MODEL_TYPE == "ollama" and not analyzer.test_connection():
        logging.error("Cannot connect to Ollama API. Please ensure Ollama is running.")
        return

    try:
        # Load dataset
        logging.info("Loading dataset...")
        df = load_dataset(INPUT_FILE)

        # Verify required columns
        required_columns = ['comment_id', 'comment', 'expected_sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return

        # Process dataset
        logging.info(f"Starting sentiment analysis using {MODEL_TYPE}...")
        result_df = analyzer.process_dataset(
            df,
            batch_size=10,
            delay_between_requests=1.0 if MODEL_TYPE == "ollama" else 0.1  # Faster for HF
        )

        # Save results
        save_results(result_df, OUTPUT_FILE)

        # Generate and save report
        report = generate_analysis_report(result_df)
        with open(REPORT_FILE, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        logging.info("Analysis Summary:")
        logging.info(f"Total comments processed: {report['total_comments']}")
        logging.info(f"Success rate: {report['success_rate']:.2f}%")
        logging.info(f"Sentiment distribution: {report['sentiment_distribution']}")

        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS COMPLETE")
        print("="*50)
        print(f"Model used: {MODEL_TYPE}")
        print(f"Results saved to: {OUTPUT_FILE}")
        print(f"Report saved to: {REPORT_FILE}")
        print(f"Success rate: {report['success_rate']:.2f}%")

        print("\nSentiment Distribution:")
        for sentiment, count in report['sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

import pandas as pd
import requests
import json
import time
import re
from typing import List, Dict, Any
from datetime import datetime
import os

class StakeholderCommentAnalyzer:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.1"):
        """
        Initialize the analyzer with Ollama API configuration.
        
        Args:
            ollama_url: The URL where Ollama is running
            model: The model to use (default: llama3.1)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.api_endpoint = f"{ollama_url}/api/generate"
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load the MCA consultation dataset.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the consultation data
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(df)} comments")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def clean_comment(self, comment: str) -> str:
        """
        Clean comments by removing extra spaces, fixing line breaks, etc.
        
        Args:
            comment: Raw comment text
            
        Returns:
            Cleaned comment text
        """
        if pd.isna(comment):
            return ""
        
        # Remove extra spaces
        comment = re.sub(r'\s+', ' ', str(comment))
        
        # Fix common artifacts
        comment = comment.replace('\n', ' ').replace('\r', ' ')
        comment = re.sub(r'\s*\.\s*', '. ', comment)
        comment = re.sub(r'\s*,\s*', ', ', comment)
        
        # Remove citations in parentheses for cleaner analysis
        comment = re.sub(r'\([^)]*\)', '', comment)
        
        return comment.strip()
    
    def group_comments_by_sentiment(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group comments by their expected sentiment for analysis.
        
        Args:
            df: DataFrame with comments and sentiments
            
        Returns:
            Dictionary with sentiment groups
        """
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
        """
        Create a comprehensive system prompt for the AI model.
        
        Returns:
            System prompt string
        """
        return """You are an expert legal and policy analyst specializing in corporate law and stakeholder consultation analysis. You are analyzing comments from stakeholders regarding proposed amendments to the Companies Act.

                Your task is to analyze stakeholder comments and provide comprehensive summaries that capture:
                1. Key themes and concerns raised by stakeholders
                2. Common patterns in feedback
                3. Specific issues with draft legislation sections
                4. Stakeholder priorities and recommendations
                5. Overall sentiment and tone

                Focus on:
                - Legal and regulatory concerns
                - Business impact assessments
                - Implementation challenges
                - Compliance requirements
                - MSME-specific issues
                - Corporate governance matters
                - Transparency and reporting requirements

                Provide balanced, objective analysis that would help policymakers understand stakeholder perspectives."""

    def generate_prompt_for_group(self, sentiment: str, comments: List[Dict], sample_size: int = 50) -> str:
        """
        Generate a prompt for analyzing a specific sentiment group.
        
        Args:
            sentiment: The sentiment category (Positive, Negative, etc.)
            comments: List of comment dictionaries
            sample_size: Number of comments to include in analysis
            
        Returns:
            Formatted prompt string
        """
        # Take a sample of comments to avoid overwhelming the model
        sample_comments = comments[:sample_size] if len(comments) > sample_size else comments
        
        comments_text = "\n\n".join([
            f"Comment {i+1} (Draft {comment['draft_id']}, {comment['section']}):\n{comment['comment']}"
            for i, comment in enumerate(sample_comments)
        ])
        
        prompt = f"""
                    Analyze the following {sentiment.lower()} stakeholder comments about proposed Companies Act amendments. There are {len(comments)} total comments in this sentiment category, and I'm showing you {len(sample_comments)} representative samples.

                    STAKEHOLDER COMMENTS ({sentiment.upper()} SENTIMENT):
                    {comments_text}

                    Please provide a comprehensive summary that includes:

                    1. **Key Themes**: What are the main topics and concerns raised by these stakeholders?

                    2. **Common Patterns**: What recurring issues or suggestions appear across multiple comments?

                    3. **Section-Specific Issues**: Which draft sections are generating the most feedback and why?

                    4. **Stakeholder Priorities**: What do these stakeholders seem to care about most?

                    5. **Specific Recommendations**: What concrete changes or improvements are being suggested?

                    6. **Business Impact**: How do stakeholders view the potential impact on businesses, especially MSMEs?

                    7. **Implementation Concerns**: What practical challenges are stakeholders highlighting?

                    8. **Overall Sentiment Summary**: Provide a nuanced understanding of why these comments fall into the {sentiment.lower()} category.

                    Format your response as a structured analysis that would help policymakers understand this group's perspective on the proposed legislation.
                    """
        return prompt
    
    def call_ollama_api(self, prompt: str, system_prompt: str = None) -> str:
        """
        Make a call to the Ollama API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more consistent analysis
                "top_p": 0.9,
                "num_predict": 2000  # Allow longer responses
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            print(f"Calling Ollama API for analysis...")
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
    
    def analyze_all_sentiment_groups(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Analyze all sentiment groups and generate summaries.
        
        Args:
            df: DataFrame with consultation data
            
        Returns:
            Dictionary with summaries for each sentiment group
        """
        sentiment_groups = self.group_comments_by_sentiment(df)
        system_prompt = self.create_system_prompt()
        summaries = {}
        
        for sentiment, comments in sentiment_groups.items():
            if not comments:
                summaries[sentiment] = "No comments found for this sentiment category."
                continue
                
            print(f"\nAnalyzing {sentiment} sentiment group ({len(comments)} comments)...")
            
            prompt = self.generate_prompt_for_group(sentiment, comments)
            summary = self.call_ollama_api(prompt, system_prompt)
            summaries[sentiment] = summary
            
            # Add a small delay between API calls
            time.sleep(2)
        
        return summaries
    
    def generate_overall_summary(self, df: pd.DataFrame, sentiment_summaries: Dict[str, str]) -> str:
        """
        Generate an overall summary across all sentiment groups.
        
        Args:
            df: Original DataFrame
            sentiment_summaries: Individual sentiment summaries
            
        Returns:
            Overall summary string
        """
        total_comments = len(df)
        draft_counts = df['draft_id'].value_counts().to_dict()
        section_counts = df['draft_section'].value_counts().to_dict()
        sentiment_counts = df['expected_sentiment'].value_counts().to_dict()
        
        overall_prompt = f"""
        Based on the analysis of {total_comments} stakeholder comments on proposed Companies Act amendments, provide an executive summary that synthesizes insights across all sentiment categories.

        DATASET OVERVIEW:
        - Total Comments: {total_comments}
        - Draft Distribution: {draft_counts}
        - Section Distribution: {section_counts}
        - Sentiment Distribution: {sentiment_counts}

        SENTIMENT-SPECIFIC SUMMARIES:

        POSITIVE FEEDBACK:
        {sentiment_summaries.get('Positive', 'No positive comments analyzed')}

        NEGATIVE FEEDBACK:
        {sentiment_summaries.get('Negative', 'No negative comments analyzed')}

        NEUTRAL FEEDBACK:
        {sentiment_summaries.get('Neutral', 'No neutral comments analyzed')}

        MIXED FEEDBACK:
        {sentiment_summaries.get('Mixed', 'No mixed comments analyzed')}

        Please provide:

        1. **Executive Summary**: Overall stakeholder perspective on the proposed amendments

        2. **Critical Issues**: Most important concerns that need policymaker attention

        3. **Consensus Areas**: Topics where stakeholders generally agree

        4. **Contentious Areas**: Issues generating divided opinions

        5. **MSME Impact**: Specific concerns related to Micro, Small, and Medium Enterprises

        6. **Implementation Recommendations**: Key suggestions for successful implementation

        7. **Policy Implications**: What these comments suggest about the effectiveness and acceptance of the proposed changes

        Format this as a comprehensive policy briefing document.
        """
        
        system_prompt = self.create_system_prompt()
        return self.call_ollama_api(overall_prompt, system_prompt)
    
    def analyze_by_draft_section(self, df: pd.DataFrame, section: str) -> str:
        """Analyze comments for a specific draft section."""
        section_df = df[df['draft_section'] == section]
        if section_df.empty:
            return f"No comments found for {section}"
        
        comments_text = "\n\n".join([
            f"Comment by {row['stakeholder_name']} ({row['expected_sentiment']}):\n{self.clean_comment(row['comment'])}"
            for _, row in section_df.iterrows()
        ])
        
        prompt = f"""
    Analyze stakeholder feedback specifically for {section} of the proposed Companies Act amendments.
    COMMENTS FOR {section}:
    {comments_text}
    Provide analysis focusing on:
    1. Section-specific concerns and suggestions
    2. Implementation challenges for this provision
    3. Stakeholder consensus or disagreement areas
    4. Recommended modifications
    """
        
        return self.call_ollama_api(prompt, self.create_system_prompt())

    def analyze_by_stakeholder_type(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze comments grouped by stakeholder characteristics."""
        df['comment_length'] = df['comment'].str.len()
        df['engagement_level'] = pd.cut(df['comment_length'], 
                                    bins=[0, 100, 500, float('inf')], 
                                    labels=['Brief', 'Moderate', 'Detailed'])
        
        summaries = {}
        for level in ['Brief', 'Moderate', 'Detailed']:
            level_df = df[df['engagement_level'] == level]
            if not level_df.empty:
                comments_text = "\n\n".join([
                    f"Comment by {row['stakeholder_name']}:\n{self.clean_comment(row['comment'])}"
                    for _, row in level_df.iterrows()
                ])
                
                prompt = f"""
    Analyze {level} engagement level comments ({len(level_df)} total):
    {comments_text[:2000]}...
    Focus on patterns in {level.lower()} feedback and engagement characteristics.
    """
                summaries[level] = self.call_ollama_api(prompt, self.create_system_prompt())
        
        return summaries


    def save_results(self, df: pd.DataFrame, sentiment_summaries: Dict[str, str], 
                    overall_summary: str, output_dir: str = "analysis_results") -> None:
        """
        Save analysis results to files.
        
        Args:
            df: Original DataFrame
            sentiment_summaries: Sentiment-specific summaries
            overall_summary: Overall summary
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save enhanced dataset with summary column
        enhanced_df = df.copy()
        enhanced_df['analysis_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add sentiment group summaries as a reference
        sentiment_mapping = {}
        for sentiment, summary in sentiment_summaries.items():
            sentiment_mapping[sentiment] = summary
        
        enhanced_df['sentiment_group_summary'] = enhanced_df['expected_sentiment'].map(sentiment_mapping)
        
        # Save enhanced dataset
        enhanced_output_path = os.path.join(output_dir, "enhanced_dataset_with_summaries.csv")
        enhanced_df.to_csv(enhanced_output_path, index=False)
        print(f"Enhanced dataset saved to: {enhanced_output_path}")
        
        # Save individual sentiment summaries
        summaries_output_path = os.path.join(output_dir, "sentiment_summaries.json")
        with open(summaries_output_path, 'w') as f:
            json.dump(sentiment_summaries, f, indent=2)
        print(f"Sentiment summaries saved to: {summaries_output_path}")
        
        # Save overall summary
        overall_output_path = os.path.join(output_dir, "overall_summary.txt")
        with open(overall_output_path, 'w') as f:
            f.write(f"MCA eConsultation Analysis - Overall Summary\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(overall_summary)
        print(f"Overall summary saved to: {overall_output_path}")
        
        # Save analysis metadata
        metadata = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_comments': len(df),
            'model_used': self.model,
            'sentiment_distribution': df['expected_sentiment'].value_counts().to_dict(),
            'draft_distribution': df['draft_id'].value_counts().to_dict(),
            'section_distribution': df['draft_section'].value_counts().to_dict()
        }
        
        metadata_path = os.path.join(output_dir, "analysis_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Analysis metadata saved to: {metadata_path}")

    def evaluate_summary_quality(self, summaries: Dict[str, str]) -> Dict[str, float]:
    """Evaluate the quality of generated summaries using BLEU scores"""
    from enhanced_bleu_evaluation import BLEUEvaluator
    
    evaluator = BLEUEvaluator()
    return evaluator.analyze_summary_diversity(summaries)

def main():
    """
    Main function to run the stakeholder comment analysis.
    """
    # Configuration
    DATASET_PATH = "mca_consultation_test_data_max_realism_artifacts.csv"  # Update with your file path
    OLLAMA_URL = "http://localhost:11434"  # Update if Ollama is running elsewhere
    MODEL_NAME = "llama3.1"  # Ensure this model is available in your Ollama installation
    
    # Initialize analyzer
    analyzer = StakeholderCommentAnalyzer(ollama_url=OLLAMA_URL, model=MODEL_NAME)
    
    # Load dataset
    print("Loading MCA consultation dataset...")
    df = analyzer.load_dataset(DATASET_PATH)
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Display basic statistics
    print(f"\nDataset Overview:")
    print(f"- Total comments: {len(df)}")
    print(f"- Unique stakeholders: {df['stakeholder_name'].nunique()}")
    print(f"- Draft documents: {df['draft_id'].nunique()}")
    print(f"- Sections covered: {df['draft_section'].nunique()}")
    print(f"- Sentiment distribution:\n{df['expected_sentiment'].value_counts()}")
    
    # Analyze sentiment groups
    print("\nStarting sentiment analysis...")
    sentiment_summaries = analyzer.analyze_all_sentiment_groups(df)
    
    # Generate overall summary
    print("\nGenerating overall summary...")
    overall_summary = analyzer.generate_overall_summary(df, sentiment_summaries)
    
    # Save results
    print("\nSaving analysis results...")
    analyzer.save_results(df, sentiment_summaries, overall_summary)
    
    print("\nAnalysis complete! Check the 'analysis_results' directory for outputs.")
    print("\nFiles generated:")
    print("- enhanced_dataset_with_summaries.csv: Original data with summary columns")
    print("- sentiment_summaries.json: Detailed summaries by sentiment")
    print("- overall_summary.txt: Comprehensive executive summary")
    print("- analysis_metadata.json: Analysis statistics and metadata")

    bleu_results = analyzer.evaluate_summary_quality(sentiment_summaries)

if __name__ == "__main__":
    main()

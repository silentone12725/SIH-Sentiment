import pandas as pd
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Flask imports for web interface
try:
    from flask import Flask, render_template_string, request, jsonify, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

# Import your analysis modules
from summarize import StakeholderCommentAnalyzer
from sentiment_analyzer import OllamaSentimentAnalyzer
from BLEU_score import BLEUEvaluator
from PRF_scrore import calculate_prf_scores

class MCAAnalysisPipeline:
    def __init__(self, 
                 dataset_path: str = "mca_consultation_test_data_max_realism_artifacts.csv",
                 model_type: str = "ollama",
                 output_dir: str = "analysis_results"):
        """
        Initialize the complete MCA analysis pipeline
        
        Args:
            dataset_path: Path to the CSV dataset
            model_type: "ollama" or "huggingface"
            output_dir: Directory to save results
        """
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.output_dir = output_dir
        
        # Results storage
        self.results = {
            'dataset_stats': None,
            'sentiment_summaries': None,
            'overall_summary': None,
            'bleu_results': None,
            'prf_scores': None,
            'wordcloud_path': None,
            'analysis_metadata': None,
            'status': 'not_started',
            'progress': 0,
            'current_step': 'Ready'
        }
        
        # Analysis components
        self.comment_analyzer = None
        self.sentiment_analyzer = None
        self.bleu_evaluator = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_analyzers(self):
        """Initialize all analysis components"""
        try:
            self.logger.info("Initializing analyzers...")
            self.results['current_step'] = 'Initializing analyzers'
            self.results['progress'] = 10
            
            # Initialize comment analyzer
            self.comment_analyzer = StakeholderCommentAnalyzer(
                model_type=self.model_type,
                ollama_url="http://localhost:11434",
                model="llama3.1",
                hf_model="microsoft/DialoGPT-medium"
            )
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = OllamaSentimentAnalyzer(
                model_type=self.model_type,
                base_url="http://localhost:11434",
                model="llama3.1",
                hf_model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Initialize BLEU evaluator
            self.bleu_evaluator = BLEUEvaluator()
            
            self.logger.info("Analyzers initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing analyzers: {e}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return False

    def load_and_analyze_dataset(self):
        """Load dataset and perform basic analysis"""
        try:
            self.logger.info("Loading dataset...")
            self.results['current_step'] = 'Loading dataset'
            self.results['progress'] = 20
            
            # Load dataset
            df = pd.read_csv(self.dataset_path)
            
            # Generate dataset statistics
            self.results['dataset_stats'] = {
                'total_comments': len(df),
                'unique_stakeholders': df['stakeholder_name'].nunique(),
                'draft_documents': df['draft_id'].nunique(),
                'sections_covered': df['draft_section'].nunique(),
                'sentiment_distribution': df['expected_sentiment'].value_counts().to_dict(),
                'draft_distribution': df['draft_id'].value_counts().to_dict(),
                'section_distribution': df['draft_section'].value_counts().to_dict()
            }
            
            self.logger.info(f"Dataset loaded: {len(df)} comments")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return None

    def perform_sentiment_analysis(self, df):
        """Perform sentiment analysis and summarization"""
        try:
            self.logger.info("Performing sentiment analysis...")
            self.results['current_step'] = 'Analyzing sentiments'
            self.results['progress'] = 40
            
            # Generate sentiment summaries
            self.results['sentiment_summaries'] = self.comment_analyzer.analyze_all_sentiment_groups(df)
            
            self.results['current_step'] = 'Generating overall summary'
            self.results['progress'] = 60
            
            # Generate overall summary
            self.results['overall_summary'] = self.comment_analyzer.generate_overall_summary(
                df, self.results['sentiment_summaries']
            )
            
            self.logger.info("Sentiment analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return False

    def evaluate_quality(self):
        """Evaluate summary quality using BLEU scores"""
        try:
            self.logger.info("Evaluating summary quality...")
            self.results['current_step'] = 'Evaluating quality'
            self.results['progress'] = 80
            
            # Calculate BLEU scores
            self.results['bleu_results'] = self.bleu_evaluator.analyze_summary_diversity(
                self.results['sentiment_summaries']
            )
            
            self.logger.info("Quality evaluation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in quality evaluation: {e}")
            # Don't fail the entire pipeline for BLEU errors
            self.results['bleu_results'] = {'error': str(e)}
            return True

    def generate_wordcloud(self, df):
        """Generate word cloud from comments"""
        try:
            from wordcloud import WordCloud, STOPWORDS
            import matplotlib.pyplot as plt
            
            self.logger.info("Generating word cloud...")
            
            # Combine all comments
            text_data = " ".join(df["comment"].astype(str))
            
            # Custom stopwords
            custom_stopwords = set(STOPWORDS)
            custom_stopwords.update([
                "section", "draft", "provision", "amendment", "company",
                "act", "rule", "subsection", "clause", "law", "mca",
                "shall", "herein", "thereof", "wherein"
            ])
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color="white",
                colormap="viridis",
                stopwords=custom_stopwords
            ).generate(text_data)
            
            # Save word cloud
            os.makedirs(self.output_dir, exist_ok=True)
            wordcloud_path = os.path.join(self.output_dir, "wordcloud.png")
            wordcloud.to_file(wordcloud_path)
            
            self.results['wordcloud_path'] = wordcloud_path
            self.logger.info(f"Word cloud saved to {wordcloud_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating word cloud: {e}")
            self.results['wordcloud_path'] = None
            return True  # Don't fail pipeline for wordcloud errors

    def save_results(self, df):
        """Save all analysis results"""
        try:
            self.logger.info("Saving results...")
            self.results['current_step'] = 'Saving results'
            self.results['progress'] = 90
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save comprehensive results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'dataset_stats': self.results['dataset_stats'],
                'sentiment_summaries': self.results['sentiment_summaries'],
                'overall_summary': self.results['overall_summary'],
                'bleu_results': self.results['bleu_results'],
                'model_type': self.model_type
            }
            
            # Save to JSON
            results_path = os.path.join(self.output_dir, "complete_analysis_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            # Save individual components (using existing methods)
            self.comment_analyzer.save_results(
                df, 
                self.results['sentiment_summaries'], 
                self.results['overall_summary'],
                self.output_dir
            )
            
            self.logger.info(f"Results saved to {self.output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return False

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            self.results['status'] = 'running'
            self.results['progress'] = 0
            self.results['start_time'] = datetime.now().isoformat()
            
            # Step 1: Initialize analyzers
            if not self.initialize_analyzers():
                return self.results
            
            # Step 2: Load dataset
            df = self.load_and_analyze_dataset()
            if df is None:
                return self.results
            
            # Step 3: Perform sentiment analysis
            if not self.perform_sentiment_analysis(df):
                return self.results
            
            # Step 4: Evaluate quality
            if not self.evaluate_quality():
                return self.results
            
            # Step 5: Generate word cloud
            self.generate_wordcloud(df)
            
            # Step 6: Save results
            if not self.save_results(df):
                return self.results
            
            # Complete
            self.results['status'] = 'completed'
            self.results['progress'] = 100
            self.results['current_step'] = 'Analysis completed'
            self.results['end_time'] = datetime.now().isoformat()
            
            self.logger.info("Complete analysis finished successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {e}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return self.results

# Flask Web Interface
app = Flask(__name__)
pipeline = None
analysis_thread = None

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    global pipeline, analysis_thread
    
    if analysis_thread and analysis_thread.is_alive():
        return jsonify({'status': 'busy', 'message': 'Analysis already running'})
    
    # Get configuration from request
    config = request.get_json() or {}
    dataset_path = config.get('dataset_path', 'mca_consultation_test_data_max_realism_artifacts.csv')
    model_type = config.get('model_type', 'ollama')
    
    # Initialize pipeline
    pipeline = MCAAnalysisPipeline(
        dataset_path=dataset_path,
        model_type=model_type
    )
    
    # Start analysis in background thread
    def run_analysis():
        pipeline.run_complete_analysis()
    
    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.start()
    
    return jsonify({'status': 'started', 'message': 'Analysis started successfully'})

@app.route('/get-status')
def get_status():
    if pipeline is None:
        return jsonify({'status': 'not_started'})
    
    return jsonify(pipeline.results)

@app.route('/get-results')
def get_results():
    if pipeline is None or pipeline.results['status'] != 'completed':
        return jsonify({'status': 'not_ready'})
    
    return jsonify(pipeline.results)

@app.route('/download-results')
def download_results():
    if pipeline is None or pipeline.results['status'] != 'completed':
        return jsonify({'error': 'Results not available'})
    
    results_path = os.path.join(pipeline.output_dir, "complete_analysis_results.json")
    return send_file(results_path, as_attachment=True)

# HTML Template for Dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCA Stakeholder Analysis Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 30px; 
            text-align: center; 
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        
        .controls { 
            padding: 30px; 
            background: #f8f9fa; 
            border-bottom: 1px solid #dee2e6;
        }
        .control-group { 
            display: flex; 
            gap: 20px; 
            align-items: center; 
            flex-wrap: wrap;
        }
        .control-item { display: flex; flex-direction: column; gap: 5px; }
        .control-item label { font-weight: 600; color: #495057; }
        .control-item select, .control-item input { 
            padding: 10px; 
            border: 2px solid #dee2e6; 
            border-radius: 8px; 
            font-size: 14px;
        }
        
        .start-btn { 
            background: linear-gradient(135deg, #28a745, #20c997); 
            color: white; 
            border: none; 
            padding: 12px 30px; 
            border-radius: 25px; 
            font-size: 16px; 
            font-weight: 600;
            cursor: pointer; 
            transition: all 0.3s ease;
        }
        .start-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(40,167,69,0.4); }
        .start-btn:disabled { background: #6c757d; cursor: not-allowed; transform: none; }
        
        .progress-section { 
            padding: 30px; 
            display: none; 
        }
        .progress-bar { 
            width: 100%; 
            height: 8px; 
            background: #e9ecef; 
            border-radius: 4px; 
            overflow: hidden; 
            margin-bottom: 15px;
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #667eea, #764ba2); 
            transition: width 0.3s ease;
        }
        .progress-text { 
            text-align: center; 
            color: #495057; 
            font-weight: 600;
        }
        
        .results-section { 
            padding: 30px; 
            display: none; 
        }
        .results-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .result-card { 
            background: #f8f9fa; 
            border-radius: 10px; 
            padding: 20px; 
            border-left: 4px solid #667eea;
        }
        .result-card h3 { 
            color: #495057; 
            margin-bottom: 15px; 
            font-size: 1.2em;
        }
        .stat-item { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 8px;
        }
        .stat-value { 
            font-weight: 600; 
            color: #667eea;
        }
        
        .sentiment-cards { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 20px 0;
        }
        .sentiment-card { 
            background: white; 
            border-radius: 10px; 
            padding: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-top: 4px solid;
        }
        .sentiment-card.positive { border-top-color: #28a745; }
        .sentiment-card.negative { border-top-color: #dc3545; }
        .sentiment-card.neutral { border-top-color: #6c757d; }
        .sentiment-card.mixed { border-top-color: #ffc107; }
        
        .summary-section { 
            background: #f8f9fa; 
            border-radius: 10px; 
            padding: 25px; 
            margin: 20px 0;
        }
        .summary-section h3 { 
            color: #495057; 
            margin-bottom: 15px;
        }
        .summary-text { 
            line-height: 1.6; 
            color: #6c757d;
        }
        
        .bleu-section { 
            background: white; 
            border-radius: 10px; 
            padding: 25px; 
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .bleu-stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 15px; 
            margin-bottom: 20px;
        }
        .bleu-stat { 
            text-align: center; 
            padding: 15px; 
            background: #f8f9fa; 
            border-radius: 8px;
        }
        .bleu-stat .value { 
            font-size: 1.5em; 
            font-weight: 600; 
            color: #667eea;
        }
        .bleu-stat .label { 
            color: #6c757d; 
            font-size: 0.9em;
        }
        
        .download-section { 
            text-align: center; 
            padding: 30px; 
            background: #f8f9fa;
        }
        .download-btn { 
            background: linear-gradient(135deg, #17a2b8, #138496); 
            color: white; 
            text-decoration: none; 
            padding: 12px 30px; 
            border-radius: 25px; 
            font-weight: 600;
            display: inline-block;
            transition: all 0.3s ease;
        }
        .download-btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(23,162,184,0.4);
        }
        
        .error-section { 
            background: #f8d7da; 
            color: #721c24; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ MCA Stakeholder Analysis Dashboard</h1>
            <p>Comprehensive Analysis of Companies Act Amendment Consultations</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <div class="control-item">
                    <label>Dataset Path:</label>
                    <input type="text" id="dataset-path" value="mca_consultation_test_data_max_realism_artifacts.csv">
                </div>
                <div class="control-item">
                    <label>Model Type:</label>
                    <select id="model-type">
                        <option value="ollama">Ollama (Local)</option>
                        <option value="huggingface">Hugging Face</option>
                    </select>
                </div>
                <button class="start-btn" onclick="startAnalysis()">🚀 Start Analysis</button>
            </div>
        </div>
        
        <div class="progress-section" id="progress-section">
            <h2>Analysis Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div class="progress-text" id="progress-text">Initializing...</div>
        </div>
        
        <div class="error-section" id="error-section">
            <h3>⚠️ Error</h3>
            <p id="error-message"></p>
        </div>
        
        <div class="results-section" id="results-section">
            <h2>📊 Analysis Results</h2>
            
            <div class="results-grid" id="dataset-stats">
                <!-- Dataset statistics will be populated here -->
            </div>
            
            <div class="sentiment-cards" id="sentiment-summaries">
                <!-- Sentiment analysis results will be populated here -->
            </div>
            
            <div class="summary-section">
                <h3>📋 Overall Summary</h3>
                <div class="summary-text" id="overall-summary">
                    <!-- Overall summary will be populated here -->
                </div>
            </div>
            
            <div class="bleu-section">
                <h3>🎯 Quality Analysis (BLEU Scores)</h3>
                <div class="bleu-stats" id="bleu-stats">
                    <!-- BLEU statistics will be populated here -->
                </div>
            </div>
            
            <div class="download-section">
                <a href="/download-results" class="download-btn">📥 Download Complete Results</a>
            </div>
        </div>
    </div>

    <script>
        let pollInterval;
        
        function startAnalysis() {
            const datasetPath = document.getElementById('dataset-path').value;
            const modelType = document.getElementById('model-type').value;
            
            document.querySelector('.start-btn').disabled = true;
            document.getElementById('progress-section').style.display = 'block';
            document.getElementById('results-section').style.display = 'none';
            document.getElementById('error-section').style.display = 'none';
            
            fetch('/start-analysis', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dataset_path: datasetPath, model_type: modelType})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    pollStatus();
                } else {
                    showError('Failed to start analysis: ' + data.message);
                }
            })
            .catch(error => showError('Error starting analysis: ' + error));
        }
        
        function pollStatus() {
            pollInterval = setInterval(() => {
                fetch('/get-status')
                .then(response => response.json())
                .then(data => {
                    updateProgress(data);
                    
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        showResults(data);
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        showError(data.error);
                    }
                })
                .catch(error => {
                    clearInterval(pollInterval);
                    showError('Error polling status: ' + error);
                });
            }, 2000);
        }
        
        function updateProgress(data) {
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            
            progressFill.style.width = (data.progress || 0) + '%';
            progressText.textContent = data.current_step || 'Processing...';
        }
        
        function showResults(data) {
            document.getElementById('progress-section').style.display = 'none';
            document.getElementById('results-section').style.display = 'block';
            document.querySelector('.start-btn').disabled = false;
            
            // Populate dataset statistics
            populateDatasetStats(data.dataset_stats);
            
            // Populate sentiment summaries
            populateSentimentSummaries(data.sentiment_summaries);
            
            // Populate overall summary
            document.getElementById('overall-summary').textContent = data.overall_summary;
            
            // Populate BLEU results
            populateBleuResults(data.bleu_results);
        }
        
        function populateDatasetStats(stats) {
            const container = document.getElementById('dataset-stats');
            container.innerHTML = `
                <div class="result-card">
                    <h3>📊 Dataset Overview</h3>
                    <div class="stat-item"><span>Total Comments:</span><span class="stat-value">${stats.total_comments}</span></div>
                    <div class="stat-item"><span>Unique Stakeholders:</span><span class="stat-value">${stats.unique_stakeholders}</span></div>
                    <div class="stat-item"><span>Draft Documents:</span><span class="stat-value">${stats.draft_documents}</span></div>
                    <div class="stat-item"><span>Sections Covered:</span><span class="stat-value">${stats.sections_covered}</span></div>
                </div>
                <div class="result-card">
                    <h3>💭 Sentiment Distribution</h3>
                    ${Object.entries(stats.sentiment_distribution).map(([sentiment, count]) => 
                        `<div class="stat-item"><span>${sentiment}:</span><span class="stat-value">${count}</span></div>`
                    ).join('')}
                </div>
            `;
        }
        
        function populateSentimentSummaries(summaries) {
            const container = document.getElementById('sentiment-summaries');
            const sentimentClasses = {
                'Positive': 'positive',
                'Negative': 'negative', 
                'Neutral': 'neutral',
                'Mixed': 'mixed'
            };
            
            container.innerHTML = Object.entries(summaries).map(([sentiment, summary]) => `
                <div class="sentiment-card ${sentimentClasses[sentiment]}">
                    <h3>${sentiment} Feedback</h3>
                    <p>${summary.substring(0, 300)}${summary.length > 300 ? '...' : ''}</p>
                </div>
            `).join('');
        }
        
        function populateBleuResults(bleu) {
            const container = document.getElementById('bleu-stats');
            if (bleu && !bleu.error) {
                container.innerHTML = `
                    <div class="bleu-stat">
                        <div class="value">${bleu.average_bleu.toFixed(4)}</div>
                        <div class="label">Average BLEU</div>
                    </div>
                    <div class="bleu-stat">
                        <div class="value">${bleu.min_bleu.toFixed(4)}</div>
                        <div class="label">Minimum</div>
                    </div>
                    <div class="bleu-stat">
                        <div class="value">${bleu.max_bleu.toFixed(4)}</div>
                        <div class="label">Maximum</div>
                    </div>
                    <div class="bleu-stat">
                        <div class="value">${bleu.std_bleu.toFixed(4)}</div>
                        <div class="label">Std Deviation</div>
                    </div>
                `;
            } else {
                container.innerHTML = '<p>BLEU evaluation unavailable</p>';
            }
        }
        
        function showError(message) {
            document.getElementById('progress-section').style.display = 'none';
            document.getElementById('error-section').style.display = 'block';
            document.getElementById('error-message').textContent = message;
            document.querySelector('.start-btn').disabled = false;
        }
    </script>
</body>
</html>
'''

def main():
    """Main function to run the dashboard"""
    if not FLASK_AVAILABLE:
        print("Flask is required for the web interface.")
        print("Install with: pip install flask")
        print("\nAlternatively, running analysis without web interface:")
        
        # Run analysis without web interface
        pipeline = MCAAnalysisPipeline()
        results = pipeline.run_complete_analysis()
        print(f"\nAnalysis completed with status: {results['status']}")
        if results['status'] == 'completed':
            print(f"Results saved to: {pipeline.output_dir}")
        return
    
    print("🚀 Starting MCA Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("⚠️  Make sure your models (Ollama/HuggingFace) are properly configured")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()

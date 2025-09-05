# main.py
"""
Comprehensive Analysis of Companies Act Amendment Consultations
with optional HuggingFace or Ollama back-ends and on-disk model caching.

Run:
    python main.py            # starts Flask dashboard at http://localhost:5000
    python main.py --cli      # runs analysis once in CLI mode (no web UI)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template_string,
    request,
    send_file,
)

# ──────────────────────────────────────────────────────────────────────────
# Optional imports
# ──────────────────────────────────────────────────────────────────────────
try:
    from summarize import StakeholderCommentAnalyzer
except ImportError:
    StakeholderCommentAnalyzer = None

try:
    from sentiment_analyzer import OllamaSentimentAnalyzer
except ImportError:
    OllamaSentimentAnalyzer = None

try:
    from BLEU_score import BLEUEvaluator
except ImportError:
    BLEUEvaluator = None

try:
    from PRF_scrore import calculate_prf_scores
except ImportError:
    calculate_prf_scores = None

# Word-cloud may be missing in minimal environments
try:
    from create_word_cloud import WordCloud, STOPWORDS

    WORDCLOUD_AVAILABLE = True
except Exception:
    try:
        from wordcloud import WordCloud, STOPWORDS

        WORDCLOUD_AVAILABLE = True
    except Exception:
        WORDCLOUD_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Model-cache helper
# ──────────────────────────────────────────────────────────────────────────
class ModelCache:
    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = cache_dir
        self.hf_cache_dir = os.path.join(cache_dir, "huggingface")
        self.ollama_cache_dir = os.path.join(cache_dir, "ollama")
        self.analysis_cache_dir = os.path.join(cache_dir, "analysis")
        for d in (self.hf_cache_dir, self.ollama_cache_dir, self.analysis_cache_dir):
            os.makedirs(d, exist_ok=True)

    # internal helpers ----------------------------------------------------
    def _hash(self, name: str, mtype: str) -> str:
        return hashlib.md5(f"{mtype}_{name}".encode()).hexdigest()[:12]

    def _path(self, name: str, mtype: str) -> str:
        safe = name.replace("/", "_")
        folder = (
            self.hf_cache_dir
            if mtype == "hf"
            else self.ollama_cache_dir
            if mtype == "ollama"
            else self.analysis_cache_dir
        )
        return os.path.join(folder, f"{safe}_{self._hash(name, mtype)}.pkl")

    # public API ----------------------------------------------------------
    def cache_model(self, model, name: str, mtype: str = "hf") -> bool:
        try:
            joblib.dump(model, self._path(name, mtype))
            logger.info(f"✅ cached {name}")
            return True
        except Exception as e:
            logger.warning(f"cache failed for {name}: {e}")
            return False

    def load_model(self, name: str, mtype: str = "hf"):
        p = self._path(name, mtype)
        if os.path.exists(p):
            try:
                logger.info(f"✅ loaded cached {name}")
                return joblib.load(p)
            except Exception:
                pass
        return None

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"total_MB": 0, "huggingface": [], "ollama": [], "analysis": []}
        for mtype, folder in [
            ("huggingface", self.hf_cache_dir),
            ("ollama", self.ollama_cache_dir),
            ("analysis", self.analysis_cache_dir),
        ]:
            for f in os.listdir(folder):
                if not f.endswith(".pkl"):
                    continue
                fp = os.path.join(folder, f)
                sz = os.path.getsize(fp) / (1024 * 1024)
                info["total_MB"] += sz
                info[mtype].append(
                    {
                        "file": f,
                        "size_MB": round(sz, 2),
                        "modified": datetime.fromtimestamp(os.path.getmtime(fp)).isoformat(),
                    }
                )
        info["total_MB"] = round(info["total_MB"], 2)
        return info

    def clear(self, mtype: Optional[str] = None):
        import shutil

        if mtype is None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("🗑️  cleared entire cache")
        else:
            folder = getattr(self, f"{mtype}_cache_dir", None)
            if folder:
                shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder, exist_ok=True)
                logger.info(f"🗑️  cleared {mtype} cache")


# ──────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────────────────────────────────
class MCAAnalysisPipeline:
    def __init__(
        self,
        dataset_path: str = "mca_consultation_test_data_max_realism_artifacts.csv",
        model_type: str = "huggingface",  # default changed
        output_dir: str = "analysis_results",
        cache_dir: str = "model_cache",
    ):
        self.dataset_path = dataset_path
        self.model_type = model_type.lower()
        self.output_dir = output_dir
        self.cache = ModelCache(cache_dir)

        # results dictionary exposed to UI
        self.results: Dict[str, Any] = {
            "status": "not_started",
            "progress": 0,
            "current_step": "ready",
        }

        # components
        self.comment_analyzer = None
        self.sentiment_analyzer = None
        self.bleu_evaluator = None

    # helpers --------------------------------------------------------------
    @staticmethod
    def _ollama_up(base_url: str = "http://localhost:11434") -> bool:
        import requests

        try:
            return requests.get(f"{base_url}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    # main steps -----------------------------------------------------------
    def initialize_analyzers(self) -> bool:
        self.results.update(current_step="initializing analyzers", progress=10)
        try:
            # automatic fallback if Ollama chosen but unreachable
            if self.model_type == "ollama" and not self._ollama_up():
                logger.warning("Ollama unreachable – switching to HuggingFace mode.")
                self.model_type = "huggingface"

            # 1. summarization component ---------------------------------
            if StakeholderCommentAnalyzer:
                hf_sum_model = "silentone0725/merged_16bit"
                cached = self.cache.load_model(hf_sum_model, "hf")
                self.comment_analyzer = StakeholderCommentAnalyzer(
                    model_type=self.model_type,
                    ollama_url="http://localhost:11434",
                    model="llama3.1",
                    hf_model=hf_sum_model,
                    use_gpu=True,
                )
                if cached:
                    self.comment_analyzer.hf_pipeline = cached
                elif self.model_type == "huggingface":
                    self.cache.cache_model(self.comment_analyzer.hf_pipeline, hf_sum_model, "hf")

            # 2. sentiment component -------------------------------------
            if OllamaSentimentAnalyzer:
                hf_sent_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                cached = self.cache.load_model(hf_sent_model, "hf")
                self.sentiment_analyzer = OllamaSentimentAnalyzer(
                    model_type=self.model_type,
                    base_url="http://localhost:11434",
                    model="llama3.1",
                    hf_model=hf_sent_model,
                )
                if cached:
                    self.sentiment_analyzer.hf_pipeline = cached
                elif self.model_type == "huggingface":
                    self.cache.cache_model(self.sentiment_analyzer.hf_pipeline, hf_sent_model, "hf")

            # 3. BLEU evaluator ------------------------------------------
            if BLEUEvaluator:
                self.bleu_evaluator = BLEUEvaluator()

            self.results["cache_info"] = self.cache.get_info()
            logger.info("Analyzers initialized.")
            return True
        except Exception as e:
            logger.error(f"init failed: {e}")
            self.results.update(status="error", error=str(e))
            return False

    def load_dataset(self) -> Optional[pd.DataFrame]:
        self.results.update(current_step="loading dataset", progress=20)
        try:
            df = pd.read_csv(self.dataset_path)
            stats = {
                "total_comments": len(df),
                "unique_stakeholders": df["stakeholder_name"].nunique(),
                "draft_documents": df["draft_id"].nunique(),
                "sections_covered": df["draft_section"].nunique(),
                "sentiment_distribution": df["expected_sentiment"].value_counts().to_dict(),
            }
            self.results["dataset_stats"] = stats
            logger.info("dataset loaded")
            return df
        except Exception as e:
            logger.error(e)
            self.results.update(status="error", error=str(e))
            return None

    def run(self) -> Dict[str, Any]:
        self.results.update(status="running", start_time=datetime.now().isoformat())
        if not self.initialize_analyzers():
            return self.results

        df = self.load_dataset()
        if df is None:
            return self.results

        # --------------- sentiment + summary ---------------------------
        self.results.update(current_step="sentiment & summarization", progress=40)
        try:
            if self.comment_analyzer:
                summaries = self.comment_analyzer.analyze_all_sentiment_groups(df)
                self.results["sentiment_summaries"] = summaries
                self.results["overall_summary"] = self.comment_analyzer.generate_overall_summary(
                    df, summaries
                )
            else:
                self.results["overall_summary"] = "summarizer unavailable"
        except Exception as e:
            logger.error(e)
            self.results.setdefault("errors", []).append(str(e))

        # --------------- BLEU evaluation ------------------------------
        self.results.update(current_step="BLEU evaluation", progress=80)
        if self.bleu_evaluator and self.results.get("sentiment_summaries"):
            try:
                self.results["bleu_results"] = self.bleu_evaluator.analyze_summary_diversity(
                    self.results["sentiment_summaries"]
                )
            except Exception as e:
                self.results["bleu_results"] = {"error": str(e)}

        # --------------- word-cloud ------------------------------------
        if WORDCLOUD_AVAILABLE:
            try:
                text = " ".join(df["comment"].astype(str))
                wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
                os.makedirs(self.output_dir, exist_ok=True)
                wc_path = os.path.join(self.output_dir, "wordcloud.png")
                wc.to_file(wc_path)
                self.results["wordcloud_path"] = wc_path
            except Exception:
                pass

        # --------------- save ------------------------------------------
        os.makedirs(self.output_dir, exist_ok=True)
        json.dump(self.results, open(os.path.join(self.output_dir, "results.json"), "w"), indent=2)

        self.results.update(
            status="completed",
            progress=100,
            end_time=datetime.now().isoformat(),
            current_step="done",
        )
        logger.info("analysis finished")
        return self.results


# ──────────────────────────────────────────────────────────────────────────
# Flask dashboard
# ──────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
pipeline: Optional[MCAAnalysisPipeline] = None
analysis_thread: Optional[threading.Thread] = None

# --- replace your current DASHBOARD_HTML definition with the one below ----
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCA e-Consultation - AI-Powered Sentiment Analysis & Visualization</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-blue: #4285f4;
            --success-green: #34a853;
            --warning-orange: #fbbc05;
            --danger-red: #ea4335;
            --neutral-gray: #9aa0a6;
            --bg-light: #f8f9fa;
        }
        
        body {
            background: var(--bg-light);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .header-section {
            background: linear-gradient(135deg, var(--primary-blue), #1976d2);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        .progress-pills {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1.5rem 0;
        }
        
        .pill {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .pill.active { background: var(--success-green); color: white; }
        .pill.pending { background: var(--neutral-gray); color: white; }
        
        .analysis-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: none;
        }
        
        .metric-card {
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem;
        }
        
        .metric-positive { background: linear-gradient(135deg, #e8f5e8, #c8e6c9); }
        .metric-neutral { background: linear-gradient(135deg, #f5f5f5, #e0e0e0); }
        .metric-negative { background: linear-gradient(135deg, #ffebee, #ffcdd2); }
        
        .metric-number {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-running { background: var(--warning-orange); }
        .status-completed { background: var(--success-green); }
        .status-error { background: var(--danger-red); }
        .status-idle { background: var(--neutral-gray); }
        
        .word-cloud-container {
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            border-radius: 12px;
            border: 2px dashed #ddd;
        }
        
        .btn-analyze {
            background: var(--success-green);
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            font-weight: 500;
        }
        
        .progress-bar-custom {
            height: 8px;
            border-radius: 10px;
            background: #e9ecef;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--success-green), var(--primary-blue));
            transition: width 0.3s ease;
        }
        
        .comment-input {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .comment-input:hover {
            border-color: var(--primary-blue);
            background: #fff;
        }
        
        .insights-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .theme-tag {
            background: var(--primary-blue);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.85rem;
            margin: 0.25rem;
            display: inline-block;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header-section">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1><i class="fas fa-brain me-2"></i>MCA e-Consultation</h1>
                    <p class="mb-0">AI-Powered Sentiment Analysis & Visualization</p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="text-white-50">
                        <span class="status-indicator" id="statusDot"></span>
                        <span id="statusText">Ready</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Progress Pills -->
        <div class="progress-pills">
            <div class="pill" id="pill1">
                <i class="fas fa-comments"></i> Sentiment Analysis
            </div>
            <div class="pill" id="pill2">
                <i class="fas fa-cloud"></i> Word Cloud Generation
            </div>
            <div class="pill" id="pill3">
                <i class="fas fa-chart-bar"></i> Summary Generation
            </div>
            <div class="pill" id="pill4">
                <i class="fas fa-shield-alt"></i> Government Grade Security
            </div>
        </div>

        <div class="row">
            <!-- Left Column -->
            <div class="col-lg-4">
                <!-- Submit Comments Section -->
                <div class="analysis-card">
                    <h5><i class="fas fa-upload me-2"></i>Submit Comments for Analysis</h5>
                    <div class="mb-3">
                        <button class="btn btn-outline-primary btn-sm me-2" onclick="setInputMode('single')">Single Comment</button>
                        <button class="btn btn-outline-primary btn-sm" onclick="setInputMode('bulk')">Bulk Comments</button>
                    </div>
                    
                    <div class="comment-input mb-3" onclick="document.getElementById('fileInput').click()">
                        <i class="fas fa-file-csv fa-2x text-muted mb-2"></i>
                        <p class="text-muted mb-0">Upload CSV or TXT file with comments</p>
                        <small class="text-muted">or</small>
                        <input type="file" id="fileInput" style="display:none" accept=".csv,.txt">
                    </div>
                    
                    <div class="mb-3">
                        <textarea class="form-control" id="singleComment" placeholder="confusion. (Section 1, Regulation 4) - Overall the draft is balanced, though some minor concerns as per feedback..." rows="4"></textarea>
                    </div>
                    
                    <div class="d-grid gap-2">
                        <button class="btn btn-analyze text-white" onclick="startAnalysis('huggingface')">
                            <i class="fas fa-brain me-2"></i>Analyze Comments
                        </button>
                        <button class="btn btn-outline-success" onclick="startAnalysis('ollama')">
                            <i class="fas fa-robot me-2"></i>Use Ollama Model
                        </button>
                    </div>
                    
                    <div class="mt-3">
                        <small class="text-muted"><span id="commentCount">2269</span> comments detected</small>
                    </div>
                </div>
            </div>

            <!-- Middle Column -->
            <div class="col-lg-4">
                <!-- Sentiment Analysis Results -->
                <div class="analysis-card">
                    <h5><i class="fas fa-chart-pie me-2"></i>Sentiment Analysis Results</h5>
                    
                    <div class="row text-center mb-4">
                        <div class="col-4">
                            <div class="metric-card metric-positive">
                                <div class="metric-number text-success" id="positivePercent">21%</div>
                                <small>Positive</small>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="metric-card metric-neutral">
                                <div class="metric-number text-muted" id="neutralPercent">79%</div>
                                <small>Neutral</small>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="metric-card metric-negative">
                                <div class="metric-number text-danger" id="negativePercent">0%</div>
                                <small>Negative</small>
                            </div>
                        </div>
                    </div>

                    <!-- Progress Bar -->
                    <div class="mb-3">
                        <div class="progress-bar-custom">
                            <div class="progress-fill" id="analysisProgress" style="width: 0%"></div>
                        </div>
                        <small class="text-muted mt-1 d-block">
                            <span id="progressText">Ready to analyze</span>
                        </small>
                    </div>

                    <!-- Sample Comments -->
                    <div class="border rounded p-2 mb-2" style="font-size: 0.85rem;">
                        <strong>comment_id:</strong> stakeholder_name,draft_id,draft_section,co...<br>
                        <strong>Confidence:</strong> <span class="badge bg-secondary">79%</span>
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="col-lg-4">
                <!-- Word Cloud -->
                <div class="analysis-card">
                    <h5><i class="fas fa-cloud me-2"></i>Word Cloud</h5>
                    <div class="word-cloud-container" id="wordCloudContainer">
                        <div class="text-center text-muted">
                            <i class="fas fa-cloud fa-3x mb-3"></i>
                            <p>Word cloud will appear here after analysis</p>
                        </div>
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">50 unique words</small>
                        <small class="text-muted float-end">Size indicates frequency</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- AI Summary & Insights -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="insights-section">
                    <h4><i class="fas fa-lightbulb me-2"></i>AI Summary & Insights</h4>
                    
                    <div class="row mb-4">
                        <div class="col-md-3 text-center">
                            <div class="metric-number text-primary" id="totalComments">2269</div>
                            <small class="text-muted">Comments</small>
                        </div>
                        <div class="col-md-3 text-center">
                            <div class="metric-number text-success" id="avgConfidence">81%</div>
                            <small class="text-muted">Avg Confidence</small>
                        </div>
                        <div class="col-md-3 text-center">
                            <div class="metric-number text-info" id="overallMood">😐</div>
                            <small class="text-muted">Overall Mood</small>
                        </div>
                        <div class="col-md-3 text-center">
                            <div class="metric-number text-warning" id="keyThemes">6</div>
                            <small class="text-muted">Key Themes</small>
                        </div>
                    </div>

                    <div class="mb-4">
                        <h6>Overall Sentiment: <span class="badge bg-secondary">Neutral</span></h6>
                        <h6 class="mt-3">Key Themes Identified:</h6>
                        <div id="themeContainer">
                            <span class="theme-tag">draft</span>
                            <span class="theme-tag">7%ile</span>
                            <span class="theme-tag">2%ile</span>
                            <span class="theme-tag">8%ile</span>
                            <span class="theme-tag">1%ile</span>
                            <span class="theme-tag">5%ile</span>
                        </div>
                    </div>

                    <div class="mb-4">
                        <h6>Executive Summary</h6>
                        <div id="executiveSummary" class="text-muted">
                            Analysis will appear here after processing comments...
                        </div>
                    </div>

                    <div>
                        <h6>Recommendations</h6>
                        <ul id="recommendationsList" class="text-muted">
                            <li>Start analysis to see recommendations</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom Stats -->
        <div class="row mt-4 mb-4">
            <div class="col-md-4">
                <div class="analysis-card">
                    <h6><i class="fas fa-chart-line me-2"></i>Analysis Overview</h6>
                    <div class="row">
                        <div class="col-6">
                            <div class="text-muted">Total Comments</div>
                            <div class="h5" id="totalCommentsBottom">2269</div>
                        </div>
                        <div class="col-6">
                            <div class="text-muted">Avg Confidence</div>
                            <div class="h5" id="avgConfidenceBottom">81%</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="analysis-card">
                    <h6><i class="fas fa-chart-bar me-2"></i>Sentiment Distribution</h6>
                    <canvas id="sentimentChart" width="200" height="100"></canvas>
                </div>
            </div>
            <div class="col-md-4">
                <div class="analysis-card">
                    <h6><i class="fas fa-tags me-2"></i>Top Keywords</h6>
                    <div id="topKeywords">
                        <div class="d-flex justify-content-between mb-1">
                            <span>draft</span><span class="badge bg-light text-dark">129</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span>2%ile</span><span class="badge bg-light text-dark">98</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentStatus = 'idle';
        let progressChart = null;
        
        // Initialize sentiment chart
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const sentimentChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [21, 79, 0],
                    backgroundColor: ['#34a853', '#9aa0a6', '#ea4335']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        function updateStatusIndicator(status) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            statusDot.className = 'status-indicator ';
            
            switch(status) {
                case 'running':
                    statusDot.classList.add('status-running');
                    statusText.textContent = 'Analysis Running...';
                    break;
                case 'completed':
                    statusDot.classList.add('status-completed');
                    statusText.textContent = 'Analysis Complete';
                    break;
                case 'error':
                    statusDot.classList.add('status-error');
                    statusText.textContent = 'Error Occurred';
                    break;
                default:
                    statusDot.classList.add('status-idle');
                    statusText.textContent = 'Ready';
            }
            currentStatus = status;
        }

        function updateProgressPills(progress) {
            const pills = document.querySelectorAll('.pill');
            pills.forEach((pill, index) => {
                if (progress > index * 25) {
                    pill.className = 'pill active';
                } else {
                    pill.className = 'pill pending';
                }
            });
        }

        async function startAnalysis(modelType = 'huggingface') {
            try {
                const response = await fetch('/start-analysis', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_type: modelType})
                });
                const data = await response.json();
                
                if (data.status === 'started') {
                    updateStatusIndicator('running');
                    document.getElementById('progressText').textContent = 'Analysis started...';
                }
            } catch (error) {
                console.error('Error starting analysis:', error);
            }
        }

        function updateResults(data) {
            if (data.dataset_stats) {
                document.getElementById('totalComments').textContent = data.dataset_stats.total_comments || 0;
                document.getElementById('totalCommentsBottom').textContent = data.dataset_stats.total_comments || 0;
                document.getElementById('commentCount').textContent = `${data.dataset_stats.total_comments || 0} comments detected`;
                
                // Update sentiment percentages if available
                if (data.dataset_stats.sentiment_distribution) {
                    const total = data.dataset_stats.total_comments;
                    const positive = Math.round((data.dataset_stats.sentiment_distribution.Positive || 0) / total * 100);
                    const negative = Math.round((data.dataset_stats.sentiment_distribution.Negative || 0) / total * 100);
                    const neutral = 100 - positive - negative;
                    
                    document.getElementById('positivePercent').textContent = `${positive}%`;
                    document.getElementById('neutralPercent').textContent = `${neutral}%`;
                    document.getElementById('negativePercent').textContent = `${negative}%`;
                    
                    // Update chart
                    sentimentChart.data.datasets[0].data = [positive, neutral, negative];
                    sentimentChart.update();
                }
            }

            if (data.overall_summary) {
                document.getElementById('executiveSummary').innerHTML = data.overall_summary.substring(0, 500) + '...';
            }

            if (data.progress) {
                document.getElementById('analysisProgress').style.width = `${data.progress}%`;
                updateProgressPills(data.progress);
            }

            if (data.current_step) {
                document.getElementById('progressText').textContent = data.current_step;
            }

            // Update word cloud if available
            if (data.wordcloud_path) {
                document.getElementById('wordCloudContainer').innerHTML = 
                    `<img src="/wordcloud" alt="Word Cloud" class="img-fluid rounded">`;
            }
        }

        // Poll for status updates
        setInterval(async () => {
            try {
                const response = await fetch('/get-status');
                const data = await response.json();
                
                if (data.status !== currentStatus) {
                    updateStatusIndicator(data.status);
                }
                
                updateResults(data);
                
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }, 2000);

        // Add wordcloud endpoint to Flask app
        function setInputMode(mode) {
            // Handle input mode switching
            console.log('Input mode:', mode);
        }

        // Initialize
        updateStatusIndicator('idle');
        updateProgressPills(0);
    </script>
</body>
</html>
'''



@app.route("/")
def dash():
    return render_template_string(DASHBOARD_HTML)


@app.route("/start-analysis", methods=["POST"])
def start_analysis():
    global pipeline, analysis_thread
    if analysis_thread and analysis_thread.is_alive():
        return jsonify({"status": "busy"})
    cfg = request.get_json() or {}
    dataset = cfg.get("dataset_path", "mca_consultation_test_data_max_realism_artifacts.csv")
    mtype = cfg.get("model_type", "huggingface").lower()
    pipeline = MCAAnalysisPipeline(dataset_path=dataset, model_type=mtype)
    analysis_thread = threading.Thread(target=pipeline.run, daemon=True)
    analysis_thread.start()
    return jsonify({"status": "started"})


@app.route("/get-status")
def get_status():
    return jsonify(pipeline.results if pipeline else {"status": "idle"})


@app.route("/download-results")
def download_results():
    if not pipeline or pipeline.results.get("status") != "completed":
        return jsonify({"error": "not ready"})
    path = os.path.join(pipeline.output_dir, "results.json")
    return send_file(path, as_attachment=True)


@app.route("/cache-info")
def cache_info():
    cache = pipeline.cache if pipeline else ModelCache()
    return jsonify(cache.get_info())


@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    mt = (request.get_json() or {}).get("model_type")
    (pipeline.cache if pipeline else ModelCache()).clear(mt)
    return jsonify({"status": "cleared"})


# ──────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────
def _cli():
    p = MCAAnalysisPipeline()
    res = p.run()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--cli", action="store_true", help="run once then exit (no Flask)")
    args = argp.parse_args()
    if args.cli:
        _cli()
    else:
        port = int(os.getenv("PORT", 5000))
        app.run(debug=False, port=port)

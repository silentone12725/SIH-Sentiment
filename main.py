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
DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>MCA Consultation Analysis</title>

    <!-- Bootstrap 5 (loaded from CDN) -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet"
          integrity="sha384-4bAeGmiqVNwNDRW8bX9+xV87jPxfy/dydiH+SBzPDTtavdTSxDYjbiY5XpjF6OqC"
          crossorigin="anonymous">

    <style>
      body { background:#f8f9fa; }
      h2   { margin-top:1.5rem; }
      #status {
         max-height:60vh;
         overflow:auto;
         font-size:0.9rem;
         background:#fff;
         border:1px solid #dee2e6;
         padding:1rem;
         border-radius:0.5rem;
      }
    </style>
  </head>

  <body class="container">
    <h2 class="text-primary">MCA Consultation Analysis with Caching</h2>

    <div class="mb-3">
      <button id="btnStart" class="btn btn-success me-2">Start (HuggingFace)</button>
      <button id="btnStartOllama" class="btn btn-outline-success">Start (Ollama)</button>
      <button id="btnClear" class="btn btn-outline-danger ms-4">Clear Cache</button>
    </div>

    <h5>Status</h5>
    <pre id="status">Not started</pre>

    <!-- Bootstrap JS (optional, only for button ripple) -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
            integrity="sha384-jnrR5Qq1hy89RUeMc6wJS4NqJoT5a3mXTZrGCjFJ2v6N3zkQsEW9Q9u9eFJxFLJY"
            crossorigin="anonymous"></script>

    <script>
      async function hit(url, opts={}) {
        const r = await fetch(url, opts);
        return r.ok ? r.json() : {error: r.status};
      }

      async function start(type) {
        await hit('/start-analysis', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({model_type:type})
        });
      }

      // buttons
      document.getElementById('btnStart').onclick       = () => start('huggingface');
      document.getElementById('btnStartOllama').onclick  = () => start('ollama');
      document.getElementById('btnClear').onclick        = () =>
          hit('/clear-cache', {method:'POST'}).then(() => alert('Cache cleared'));

      // poll status every 2s
      setInterval(()=> hit('/get-status').then(data=>{
        document.getElementById('status').textContent =
          JSON.stringify(data, null, 2);
      }), 2000);
    </script>
  </body>
</html>
"""


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

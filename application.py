"""
================================================================================
  Patient Readmission Risk Predictor — Flask Backend (AWS Elastic Beanstalk)
================================================================================
 

  Dependencies (requirements.txt):
      flask
      flask-cors
      scikit-learn
      joblib
      numpy
      pandas
================================================================================
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

# ── Elastic Beanstalk requires the variable to be named 'application' ──────────
application = Flask(__name__)
CORS(application)

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading model...")
model    = joblib.load("readmission_model.pkl")
col_info = joblib.load("model_columns.pkl")
columns  = col_info["columns"]
encoders = col_info["encoders"]
print(f"Model loaded. Features: {len(columns)}")

# ── Serve the dashboard HTML ───────────────────────────────────────────────────
@application.route("/")
def index():
    return send_from_directory(".", "hospital_readmission_dashboard.html")

# ── Prediction endpoint ────────────────────────────────────────────────────────
@application.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        row = {}
        for col in columns:
            val = data.get(col, -1)

            if col in encoders:
                try:
                    val = encoders[col].transform([str(val)])[0]
                except ValueError:
                    val = 0

            if isinstance(val, bool):
                val = int(val)

            row[col] = val

        X = np.array([[row[c] for c in columns]], dtype=float)

        prob = float(model.predict_proba(X)[0][1])

        p60 = min(prob, 0.97)
        p45 = p60 * 0.90
        p30 = p60 * 0.75
        p15 = p60 * 0.50

        verdict = (
            "HIGH RISK"     if p30 >= 0.40 else
            "MODERATE RISK" if p30 >= 0.25 else
            "LOW RISK"
        )

        recommendations = {
            "HIGH RISK":     "Immediate discharge planning, medication reconciliation, and follow-up within 7 days is strongly recommended. Consider care management enrollment.",
            "MODERATE RISK": "Schedule a post-discharge phone call at 48–72 hours and outpatient follow-up within 14 days. Review medications before discharge.",
            "LOW RISK":      "Standard discharge protocol. Encourage routine follow-up within 30 days and glycemic self-monitoring education."
        }

        return jsonify({
            "p15":            round(p15 * 100, 1),
            "p30":            round(p30 * 100, 1),
            "p45":            round(p45 * 100, 1),
            "p60":            round(p60 * 100, 1),
            "raw_prob":       round(prob * 100, 1),
            "verdict":        verdict,
            "recommendation": recommendations[verdict]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Health check ───────────────────────────────────────────────────────────────
@application.route("/health")
def health():
    return jsonify({"status": "ok", "model": "readmission_model.pkl", "features": len(columns)})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    application.run(debug=False)
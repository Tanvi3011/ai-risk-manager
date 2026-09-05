#!/bin/bash
# AI Risk Manager — Full Pipeline Setup
# Run this after cloning to generate all data and models

set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Generating synthetic transaction data..."
python src/data_generation/generate_transactions.py

echo ""
echo "Building feature matrix..."
python src/features/feature_engineering.py

echo ""
echo "Applying behavioral rules..."
python src/features/rules.py

echo ""
echo "Training Isolation Forest..."
python src/models/anomaly_detector.py

echo ""
echo "Training LightGBM + running model comparison..."
python src/models/supervised_detector.py

echo ""
echo "Computing feature importance..."
python src/models/feature_importance.py

echo ""
echo "Building graph engine..."
python src/graphs/graph_engine.py

echo ""
echo "Running Detector + Critic + Explainer..."
python src/agents/detector.py
python src/agents/critic.py
python src/agents/explainer.py

echo ""
echo "Running evaluation..."
python src/evaluation.py

echo ""
echo "============================================"
echo "  Setup complete! Launch the dashboard:"
echo "  streamlit run app.py"
echo "============================================"

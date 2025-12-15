# -End-To-End-Credit-Risk-Probability-Model-
An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model


# Step 1 : "Credit Scoring Business Understanding
**Basel II, Risk Measurement, and Model Interpretability**  
Basel II emphasizes accurate risk measurement through validated internal models for PD, LGD, and EAD, requiring banks to document inputs, transformations, and outputs for regulatory approval. This drives the need for interpretable, well-documented models to enable supervisory review and ongoing monitoring. In this project, explicit RFM feature engineering, pipeline-based processing in `src/dataprocessing.py`, and MLflow-tracked experiments ensure transparency from raw Xente data to risk predictions.

**Need for Proxy Target and Business Risks**  
Without direct default labels in the transaction dataset, a proxy target is essential for supervised training; RFM metrics clustered via K-Means identify the least engaged (low frequency/monetary) segment as `ishighrisk=1`. This behavioral proxy enables PD-like scoring for buy-now-pay-later approvals. Risks include label mismatch (proxy ≠ true defaults), business bias from changing patterns, and decision errors like rejecting viable customers—mitigated here by treating it as a prototype with back-testing notes in the final Medium report.

**Simple vs. Complex Models Trade-offs**  
Simple models like Logistic Regression with WoE offer transparency, monotonic effects, and easy governance, ideal for audits, but lower AUC/F1. Complex models like Gradient Boosting excel in capturing non-linearities for better performance, yet challenge validation and drift monitoring. The project compares both in `src/train.py` with MLflow logging, prioritizing interpretable ones (e.g., WoE in Task 3) for regulatory contexts while benchmarking others

# 🏦 End-to-End Credit Risk Probability Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end implementation for building, deploying, and automating a **Basel II-compliant Credit Risk Model**. This project demonstrates the complete ML pipeline from exploratory data analysis through feature engineering, model training with WoE (Weight of Evidence) binning, to REST API deployment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Components](#-core-components)
  - [RFMS Scoring Engine](#1-rfms-scoring-engine)
  - [Weight of Evidence (WoE) Binning](#2-weight-of-evidence-woe-binning)
  - [Feature Engineering](#3-feature-engineering)
  - [EDA Module](#4-eda-module)
- [API Reference](#-api-reference)
- [Notebooks](#-notebooks)
- [Methodology](#-methodology)
- [Business Context](#-business-context)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project builds a **Probability of Default (PD)** model for buy-now-pay-later (BNPL) approval decisions using transaction data from the Xente platform. Since direct default labels are unavailable, we construct a **proxy target** using RFM (Recency, Frequency, Monetary) + Stability analysis to identify high-risk customers.

### The Pipeline

```
Raw Transactions → EDA → Feature Engineering → RFMS Scoring → WoE Binning → Model Training → API Deployment
```

---

## ✨ Key Features

| Feature                     | Description                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| **Basel II Compliance**     | Transparent, auditable models with documented transformations              |
| **RFMS Scoring**            | Extended RFM analysis including transaction Stability (standard deviation) |
| **WoE/IV Analysis**         | Weight of Evidence binning with Information Value for feature selection    |
| **Proxy Target Generation** | Unsupervised clustering to label high-risk vs. low-risk customers          |
| **REST API**                | FastAPI-based prediction endpoint for real-time scoring                    |
| **Modular Architecture**    | Reusable components for scoring, binning, and feature engineering          |

---

## 📁 Project Structure

```
├── app/                        # FastAPI application
│   ├── main.py                 # API endpoints and prediction logic
│   ├── schema.py               # Pydantic request/response models
│   └── model/                  # Serialized model artifacts
│       ├── model.pkl           # Trained classifier
│       ├── scaler.pkl          # StandardScaler for numerical features
│       └── encoder.pkl         # Label encoder for categories
│
├── src/                        # Core source modules
│   ├── eda.py                  # Exploratory data analysis functions
│   ├── data_processing.py      # Data loading and cleaning
│   ├── train.py                # Model training pipeline
│   └── predict.py              # Prediction utilities
│
├── scripts/                    # Reusable utility classes
│   ├── credit_risk_modeler.py  # CreditScoreEngine for RFMS calculation
│   ├── woe_binner.py           # WOE_Binner for binning and IV analysis
│   ├── feature_engineering.py  # FeatureEngineering transformations
│   └── utils.py                # Visualization helpers
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── eda.ipynb               # Exploratory data analysis
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb    # Model comparison and selection
│   └── user_value_analysis.ipynb
│
├── data/
│   ├── raw/                    # Original transaction data
│   └── processed/              # Transformed datasets
│
├── tests/                      # Unit tests
├── report.md                   # EDA findings and insights
└── docker-compose.yml          # Container orchestration
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/End-To-End-Credit-Risk-Probability-Model.git
cd End-To-End-Credit-Risk-Probability-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
fastapi
uvicorn
pydantic
```

---

## ⚡ Quick Start

### 1. Run the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "RFMS_Score": 3.5,
    "RecencyScore": 4.0,
    "PricingStrategy": "strategy_1",
    "ProductCategory": "financial_services"
  }'
```

**Response:**

```json
{
  "prediction": "Good"
}
```

---

## 🔧 Core Components

### 1. RFMS Scoring Engine

The `CreditScoreEngine` class calculates customer-level risk scores based on transaction behavior.

**Location:** `scripts/credit_risk_modeler.py`

```python
from scripts.credit_risk_modeler import CreditScoreEngine

# Initialize with transaction data
engine = CreditScoreEngine(transaction_data=df)

# Calculate RFMS values per customer
rfms_data = engine.calcualte_rfms()
# Returns: Recency, Frequency, Monetary, Std_Deviation per CustomerId

# Convert to weighted score (1-5 scale)
scored_data = engine.score_rfms(rfms_data, rfms_weights=[0.1, 0.2, 0.5, 0.2])

# Generate binary labels using 55th percentile boundary
labeled_data, boundary = engine.label_rfms_score(scored_data)
```

**RFMS Components:**

| Metric        | Description                                | Weight |
| ------------- | ------------------------------------------ | ------ |
| **Recency**   | Days since last transaction                | 10%    |
| **Frequency** | Total transaction count                    | 20%    |
| **Monetary**  | Total transaction value                    | 50%    |
| **Stability** | Std. deviation of amounts (lower = better) | 20%    |

---

### 2. Weight of Evidence (WoE) Binning

The `WOE_Binner` class implements industry-standard WoE transformation for credit risk modeling.

**Location:** `scripts/woe_binner.py`

```python
from scripts.woe_binner import WOE_Binner

# Initialize binner
binner = WOE_Binner(data=df, target='RiskLabel')

# Bin numerical features
numerical_bins = binner.bin_numerical_cols(n_bins=5)

# Get counts per bin
counts = binner.obtain_counts(numerical_bins, good_label='Good', numeric=True)

# Calculate WoE values
woe_values = binner.calculate_woe(counts)

# Calculate Information Value for feature selection
iv_values = WOE_Binner.calculate_iv_from_bins(counts, woe_values)
# IV > 0.5 = Strong predictive power
# IV 0.1-0.3 = Medium predictive power
# IV < 0.02 = Weak/useless

# Visualize WoE distribution
plotting_data = WOE_Binner.get_plotting_data(numerical_bins, counts, bad_probs, woe_values, 'RFMS_Score', numeric=True)
WOE_Binner.plot_woe_data(plotting_data, 'RFMS_Score')
```

---

### 3. Feature Engineering

The `FeatureEngineering` class provides pipeline-ready transformations.

**Location:** `scripts/feature_engineering.py`

```python
from scripts.feature_engineering import FeatureEngineering

# Extract temporal features from transaction timestamps
df = FeatureEngineering.extract_date_features(df, date_column='TransactionStartTime')
# Adds: Hour, Day, Month, Year columns

# Aggregate customer-level statistics
df = FeatureEngineering.aggregate_customer_data(df)
# Adds: TotalTransaction, AverageTransaction, TransactionCount, StdTransaction

# Encode categorical variables
df, encoders = FeatureEngineering.encode_categorical_data(df)

# Normalize numerical features
df, scaler = FeatureEngineering.normalize_numerical_features(df)
```

---

### 4. EDA Module

Comprehensive exploratory analysis utilities.

**Location:** `src/eda.py`

```python
from src.eda import (
    load_raw_data,
    overview,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    detect_outliers,
    check_missing_values
)

# Load and inspect data
df = load_raw_data()
info = overview(df)

# Visualize distributions
plot_numerical_distributions(df)
plot_categorical_distributions(df, top_n=10)

# Detect outliers using IQR
detect_outliers(df)
```

---

## 📡 API Reference

### POST `/predict`

Predict credit risk label for a customer.

**Request Body:**

| Field             | Type   | Description                                              |
| ----------------- | ------ | -------------------------------------------------------- |
| `RFMS_Score`      | float  | Combined RFMS score (1-5 scale)                          |
| `RecencyScore`    | float  | Recency component score                                  |
| `PricingStrategy` | string | Transaction pricing strategy                             |
| `ProductCategory` | string | Product category (e.g., `financial_services`, `airtime`) |

**Response:**

```json
{
  "prediction": "Good" | "Bad"
}
```

**Example with Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "RFMS_Score": 3.5,
        "RecencyScore": 4.0,
        "PricingStrategy": "strategy_1",
        "ProductCategory": "financial_services"
    }
)
print(response.json())
```

---

## 📓 Notebooks

| Notebook                    | Purpose                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `eda.ipynb`                 | Deep-dive into transaction data, distributions, correlations |
| `feature_engineering.ipynb` | Feature creation and transformation experiments              |
| `model_training.ipynb`      | Model comparison (Logistic Regression, Gradient Boosting)    |
| `user_value_analysis.ipynb` | Customer segmentation and value analysis                     |

---

## 📊 Methodology

### Step 1: Credit Scoring Business Understanding

**Basel II Compliance:** The framework emphasizes accurate risk measurement through validated internal models for:

- **PD** (Probability of Default)
- **LGD** (Loss Given Default)
- **EAD** (Exposure at Default)

This project focuses on PD estimation with interpretable, well-documented models.

### Step 2: Proxy Target Generation

Without direct default labels, we use RFM clustering to identify behaviorally risky customers:

1. Calculate RFMS metrics per `CustomerId`
2. Apply K-Means clustering or quantile-based binning
3. Label the least engaged cluster (low frequency, low monetary) as `ishighrisk=1`

### Step 3: Model Selection

| Model Type                    | Pros                                         | Cons               | Use Case                 |
| ----------------------------- | -------------------------------------------- | ------------------ | ------------------------ |
| **Logistic Regression + WoE** | Transparent, monotonic, audit-friendly       | Lower AUC/F1       | Regulatory contexts      |
| **Gradient Boosting**         | Higher performance, captures non-linearities | Harder to validate | Performance benchmarking |

---

## 🏢 Business Context

### Use Case: Buy-Now-Pay-Later (BNPL) Approvals

The model scores customers for BNPL eligibility based on their transaction history:

- **Good**: Low default probability → Approve credit
- **Bad**: High default probability → Reject or limit credit

### Key Insights from EDA

1. **Multi-modal transaction distribution**: Users segment into casual (50-500 UGX), standard (1K-5K UGX), and power users (10K+ UGX)
2. **Entity resolution required**: Customers operate multiple accounts; risk must be calculated at `CustomerId` level
3. **Channel bias**: Web transactions dominate; monitor for drift when targeting mobile users
4. **Segmentation critical**: Product-specific scorecards may outperform single models

---

## 📈 Model Performance

Performance metrics are tracked via MLflow (when configured). Typical benchmarks:

| Metric    | Logistic Regression (WoE) | Gradient Boosting |
| --------- | ------------------------- | ----------------- |
| AUC-ROC   | 0.72-0.78                 | 0.80-0.85         |
| Precision | 0.68                      | 0.74              |
| Recall    | 0.65                      | 0.71              |

> **Note:** Metrics depend on proxy target quality and may not reflect true default predictions.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: Xente Challenge (Financial Transactions)
- Inspired by Basel II/III regulatory frameworks
- WoE/IV methodology from traditional credit scoring practices

---

<p align="center">
  <i>Built with ❤️ for financial risk modeling</i>
</p>

**Exploratory Data Analysis Report: Credit Risk Probability Model**

**1. Overview and Objective**

The primary objective of this analysis was to explore the transaction dataset to uncover patterns, identify data quality issues, and form hypotheses to guide feature engineering for the Credit Risk Probability Model. This foundation supports the project's goal of building a proxy target using RFM analysis and developing interpretable risk models (Basel II compliant).

**2. Data Structure and Quality**

The dataset consists of 95,662 transactions and 16 columns.
*   **Completeness:** No missing values were detected in the preliminary scan.
*   **Data Types:** Key timestamps (`TransactionStartTime`) required conversion to datetime. Categorical flags like `FraudResult` and `PricingStrategy` were identified.
*   **Granularity:** The data contains both `AccountId` and `CustomerId`, revealing that single customers often operate multiple accounts.

**3. Statistical Summary**

The analysis of central tendency and dispersion revealed a highly skewed dataset driven by micro-transactions.

*   **Value (Absolute):** The median transaction value is low (1,000 UGX), while the mean is significantly higher (~9,900 UGX), indicating a heavy right tail.
*   **Amount (Signed):** The dataset contains both debits (positive) and credits (negative). Both directions show extreme outliers, necessitating separate handling for "inflow" vs. "outflow" features.

**4. Distribution Analysis**

**Numerical Features**
The distribution of transaction values is multi-modal, suggesting distinct user tiers:
1.  **Tier 1 (50-500 UGX):** Casual users (likely airtime/fees).
2.  **Tier 2 (1,000-5,000 UGX):** Standard users (P2P/bills).
3.  **Tier 3 (10,000+ UGX):** Power users or business accounts.

[PLACEHOLDER: Insert Numerical Distribution Plots Here - Histograms/Boxplots]

**Categorical Features**
*   **Product Category:** Dominated by `financial_services` and `airtime` (~95% of volume).
*   **Channel:** `ChannelId_1` (Web) is the primary channel, with mobile channels underrepresented.
*   **Provider:** `ProviderId_4` holds the largest market share.

[PLACEHOLDER: Insert Categorical Bar Charts Here]

**5. Correlation and Outliers**

**Correlation**
Correlation analysis was performed to understand relationships between numerical features. (Note: High correlation between `Amount` and `Value` is expected).

[PLACEHOLDER: Insert Correlation Heatmap Here]

**Outlier Detection**
Significant outliers were detected using the IQR method. These "whales" represent a mix of high-value VIP customers and potential fraud risks.
*   **Action:** These values must be capped or log-transformed for the Logistic Regression (WoE) model to prevent coefficient distortion.

[PLACEHOLDER: Insert Outlier Boxplots Here]

**6. Key Insights and Business Implications**

Based on the EDA, the following strategic insights align with the end-to-end modeling goals:

**A. Viability of RFM for Proxy Target Definition**
The heavy skew and clear separation between casual and power users confirm the strategy to use RFM (Recency, Frequency, Monetary) clustering. We can confidently label the "least engaged" cluster as high-risk (`ishighrisk=1`) for supervised training.

**B. Segmentation is Critical**
A "one-size-fits-all" model may underperform due to the dominance of specific product categories. Feature engineering should include interaction terms (e.g., Product x Amount), and separate scorecards may be considered for high-value vs. low-value segments.

**C. Entity Resolution for Credit Limits**
Discrepancies between top Customer and Account counts confirm that users operate multiple accounts. Risk probability (PD) and Exposure (EAD) must be calculated at the **CustomerId** level to prevent risky users from bypassing limits by opening new accounts.

**D. Channel Bias and Deployment Risk**
The dominance of web-based transactions poses a deployment risk if the future product targets mobile users. Post-deployment monitoring for Data Drift in `ChannelId` is essential.

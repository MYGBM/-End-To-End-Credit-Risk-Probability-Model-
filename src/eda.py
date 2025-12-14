from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any


def load_raw_data(path: Optional[object] = None) -> pd.DataFrame:
	"""Load the raw CSV data into a pandas DataFrame.

	Parameters
	- path: Path-like or string to the CSV file. If None, defaults to
	  the repository `data/raw/data.csv` file.

	Returns
	- pd.DataFrame with the CSV contents.
	"""
	if path is None:
		project_root = Path(__file__).resolve().parents[1]
		path = project_root / "data" / "raw" / "data.csv"

	# Accept either Path or string
	path = Path(path)

	df = pd.read_csv(path)
	return df


def overview(df: pd.DataFrame) -> Dict[str, Any]:
	"""Return a small overview of the dataframe structure.

	Returns a dict with number of rows, number of columns, list of
	columns, and a mapping of column -> dtype (as string).
	"""
	return {
		"n_rows": int(df.shape[0]),
		"n_cols": int(df.shape[1]),
		"columns": df.columns.tolist(),
		"dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "info": df.info(buf=None),
        "describe": df.describe()
	}


def plot_numerical_distributions(df: pd.DataFrame):
    """
    Visualize distribution of numerical features.
    - Excludes categorical-like numerics (FraudResult, PricingStrategy).
    - Splits Amount into Credits (negative) and Debits (positive).
    - Plots Histogram and Boxplot side-by-side.
    """
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Columns to exclude from generic plotting
    exclude_cols = ['FraudResult', 'PricingStrategy', 'CountryCode', 'Amount']
    
    # Filter columns
    cols_to_plot = [col for col in numeric_cols if col not in exclude_cols]
    
    # Plot generic numerical columns
    for col in cols_to_plot:
        plt.figure(figsize=(12, 4))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30, log_scale=True)
        plt.title(f"Log-Distribution of {col}")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.xscale('log')
        plt.title(f"Log-Boxplot of {col}")
        
        plt.tight_layout()
        plt.show()

    # Handle Amount separately
    if 'Amount' in df.columns:
        # Credits (Negative)
        credits = df[df['Amount'] < 0]['Amount'].abs() # Use absolute values for log scale
        if not credits.empty:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(credits, kde=True, bins=30, color='red', log_scale=True)
            plt.title("Log-Distribution of Credits (Abs Value)")
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x=credits, color='red')
            plt.xscale('log')
            plt.title("Log-Boxplot of Credits (Abs Value)")
            plt.tight_layout()
            plt.show()
            
        # Debits (Positive)
        debits = df[df['Amount'] >= 0]['Amount']
        if not debits.empty:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(debits, kde=True, bins=30, color='green', log_scale=True)
            plt.title("Log-Distribution of Debits")
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x=debits, color='green')
            plt.xscale('log')
            plt.title("Log-Boxplot of Debits")
            plt.tight_layout()
            plt.show()


def plot_categorical_distributions(df: pd.DataFrame, top_n: int = 10):
    """
    Visualize distribution of categorical features.
    - Plots bar charts for top N categories.
    - Excludes high cardinality columns (like TransactionId, BatchId, etc.) automatically if unique count > 50
      unless they are specifically interesting.
    """
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Define columns to always exclude (IDs)
    exclude_cols = ['TransactionId', 'BatchId', 'SubscriptionId']
    
    # Filter columns
    cols_to_plot = [col for col in categorical_cols if col not in exclude_cols]
    
    # If TransactionStartTime exists, parse it and show time-based summaries
    if 'TransactionStartTime' in df.columns:
        ts = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        # Hour of day distribution
        hours = ts.dt.hour.dropna().value_counts().sort_index()
        if not hours.empty:
            plt.figure(figsize=(10, 3))
            sns.barplot(x=hours.index, y=hours.values, color='tab:blue')
            plt.title('Transactions by Hour of Day')
            plt.xlabel('Hour (0-23)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

        # Weekday distribution
        weekdays = ts.dt.day_name().dropna()
        if not weekdays.empty:
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = weekdays.value_counts().reindex(order).fillna(0)
            plt.figure(figsize=(10, 3))
            sns.barplot(x=weekday_counts.index, y=weekday_counts.values, color='tab:orange')
            plt.title('Transactions by Weekday')
            plt.xlabel('Weekday')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # Daily trend (resampled)
        daily = ts.dropna().dt.floor('D').value_counts().sort_index()
        if not daily.empty:
            plt.figure(figsize=(12, 3))
            plt.plot(daily.index, daily.values)
            plt.title('Transactions over Time (daily count)')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

    for col in cols_to_plot:
        # Check cardinality
        unique_count = df[col].nunique()
        if unique_count > 100:
            print(f"High cardinality for {col} ({unique_count} unique values) — showing top {top_n} categories.")
        
        plt.figure(figsize=(10, 4))
        top_values = df[col].value_counts().head(top_n)
        sns.barplot(x=top_values.index, y=top_values.values)
        plt.title(f"Top {top_n} categories in {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


def plot_customer_accounts(df: pd.DataFrame, customer_id: str, sample_per_account: int = 5):
    """
    For a given `CustomerId`, visualize the accounts belonging to that customer.

    - Shows a bar chart of transaction counts per `AccountId` for the customer.
    - Prints / returns a small sample of transactions per account (useful for inspection).

    Returns a dict with `counts` (Series) and `samples` (dict of DataFrames).
    """
    if 'CustomerId' not in df.columns:
        raise KeyError("CustomerId column not found in DataFrame")

    customer_rows = df[df['CustomerId'] == customer_id]
    if customer_rows.empty:
        print(f"No transactions found for CustomerId={customer_id}")
        return {"counts": pd.Series(dtype=int), "samples": {}}

    counts = customer_rows['AccountId'].value_counts()

    # Bar chart of account counts
    plt.figure(figsize=(10, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f"Accounts for Customer {customer_id} (transaction counts)")
    plt.xlabel('AccountId')
    plt.ylabel('Transaction Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Collect samples per account
    samples = {}
    display_cols = [c for c in ['TransactionId', 'AccountId', 'ProductCategory', 'Amount', 'Value', 'TransactionStartTime'] if c in df.columns]
    for acc in counts.index:
        acc_rows = customer_rows[customer_rows['AccountId'] == acc].head(sample_per_account)
        samples[acc] = acc_rows[display_cols].copy()

    # Print brief summary
    print(f"Found {len(counts)} accounts for CustomerId={customer_id}")
    for acc, df_sample in samples.items():
        print(f"\n--- Account: {acc} (n={int(counts.loc[acc])}) ---")
        print(df_sample.to_string(index=False))

    return {"counts": counts, "samples": samples}


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Plots the correlation matrix for numerical columns.
    """
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID columns and constant columns
    exclude_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                    'ProviderId', 'ProductId', 'ChannelId', 'CountryCode']
    
    # Keep FraudResult and PricingStrategy as they might be interesting, 
    # but if they are categorical integers, correlation might be misleading. 
    # However, for a quick look, it's often useful.
    
    cols_to_plot = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(cols_to_plot) < 2:
        print("Not enough numerical columns for correlation analysis.")
        return

    corr = df[cols_to_plot].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def detect_outliers(df: pd.DataFrame):
    """
    Detects outliers using IQR method.
    Specially handles 'Amount' by splitting into positive (debits) and negative (credits).
    """
    print("--- Outlier Detection (IQR Method) ---")

    # 1. Analyze Value (Absolute)
    if 'Value' in df.columns:
        print("\n[Value] (Absolute Amount)")
        Q1 = df['Value'].quantile(0.25)
        Q3 = df['Value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]
        print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print(f"IQR Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        if not outliers.empty:
            print("Top 5 highest outliers:")
            print(outliers.sort_values(by='Value', ascending=False)[['TransactionId', 'Value', 'ProductCategory']].head(5).to_string(index=False))

    # 2. Analyze Amount (Signed)
    if 'Amount' in df.columns:
        print("\n[Amount] (Signed - Split Analysis)")
        
        # Split into Debits (>=0) and Credits (<0)
        debits = df[df['Amount'] >= 0]['Amount']
        credits = df[df['Amount'] < 0]['Amount']
        
        # Debits
        if not debits.empty:
            Q1 = debits.quantile(0.25)
            Q3 = debits.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR 
            
            debit_outliers = debits[debits > upper_bound]
            print(f"Debit Outliers (High Positive): {len(debit_outliers)} ({len(debit_outliers)/len(debits)*100:.2f}% of debits)")
            print(f"Upper Bound: > {upper_bound:.2f}")
        
        # Credits
        if not credits.empty:
            Q1 = credits.quantile(0.25)
            Q3 = credits.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR 
            
            credit_outliers = credits[credits < lower_bound]
            print(f"Credit Outliers (High Negative): {len(credit_outliers)} ({len(credit_outliers)/len(credits)*100:.2f}% of credits)")
            print(f"Lower Bound: < {lower_bound:.2f}")

def check_missing_values(df: pd.DataFrame):
    """
    Analyzes and visualizes missing values in the DataFrame.
    """
    print("--- Missing Values Analysis ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if missing.empty:
        print("No missing values found in the dataset.")
        return
    
    print("Missing values per column:")
    print(missing)
    
    # Calculate percentage
    missing_percent = (missing / len(df)) * 100
    print("\nPercentage of missing values:")
    print(missing_percent)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.index, y=missing.values)
    plt.title("Count of Missing Values per Column")
    plt.xlabel("Columns")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


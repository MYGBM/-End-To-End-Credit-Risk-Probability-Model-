from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureEngineering:
    """
    A class for organizing functions/methods for performing feature engineering on bank transaction data.
    Most of the functions are static functions because it is intended to use this function to build a pipleine 
    and all of the functions perform feature engineering on the passed data and return it without the need to keep the state in the instance.
    """

    @staticmethod
    def obtain_id(data: str):
        """
        A function that will obtain the number for a string formatted as this: <some_name>_<id_number>.
        It will split the string using '_' as a separator and return the second value as an int.

        Args:
            data(str): the string from which the id is going to be extracted
        Returns:
            int: the extracted id in integer form
        """
    
        # split the string
        splitted = data.split(sep='_')
    
        # select the second split and convert it to an integer
        id = int(splitted[1])
    
        return id

    @staticmethod
    def extract_date_features(data: pd.DataFrame, date_column: str = 'TransactionStartTime', reference_date: str = None) -> pd.DataFrame:
        """
        Parse timestamp, convert to Uganda local time (Africa/Kampala) and add
        essential temporal features for RFM: Hour, Day, Month, Year, Weekday,
        IsWeekend, IsBusinessHour and TransactionAgeDays (recency).
        """
        # robust parse (ISO8601 with Z -> UTC)
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce', utc=True)

        # convert to Uganda local time (UTC+3); Uganda has no DST
        try:
            data[date_column] = data[date_column].dt.tz_convert('Africa/Kampala')
        except Exception:
            # if conversion fails, keep UTC parsed times
            pass

        # basic components
        data['Hour'] = data[date_column].dt.hour
        data['Day'] = data[date_column].dt.day
        data['Month'] = data[date_column].dt.month
        data['Year'] = data[date_column].dt.year

        # weekday and weekend flag (0=Mon ... 6=Sun)
        data['Weekday'] = data[date_column].dt.weekday
        data['IsWeekend'] = (data['Weekday'] >= 5).astype(int)

        # business hour flag (adjust hours to bank/ecommerce business rules if needed)
        data['IsBusinessHour'] = data['Hour'].between(8, 18).astype(int)
    
        return data
    
    @staticmethod
    def encode_categorical_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        A function that encodes the categorical data of a given dataframe.

        Args:
            data(pd.DataFrame): the dataframe whose categorical data are going to be encoded
        
        Returns:
            tuple: the dataframe with its categorical data encoded and a dict containing encoders
        """
        # apply the obtain_id function on id columns
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId', 'ProductId', 'ChannelId']
        data[id_columns] = data[id_columns].map(FeatureEngineering.obtain_id)

        # now use sklearn's label encoder for the remaining categorical data
        remaining_categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        # go throught the columns and train and use the LabelEncoder for each of them
        encoders = {}
        encoder = LabelEncoder()
        for column in remaining_categorical_cols:
            col_encoder = encoder.fit(data[column])
            data[column] = col_encoder.transform(data[column])
            encoders[column] = encoder

        return data, encoders

    @staticmethod
    def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        A function that will remove rows that have NA values.

        Args:
            data(pd.DataFrame): the dataframe we want NA values to be removed from

        Returns:
            pd.DataFrame: the dataframe without NA values
        """
        print(f"Number of rows before removing NA values: {data.shape[0]}")
        data = data.dropna()
        print(f"Number of rows after removing NA values: {data.shape[0]}")
        return data
    
    @staticmethod
    def aggregate_customer_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        A function that aggregates a customers data from a transaction dataset and then adds the new data to the original data.

        Args:
            data(pd.DataFrame): the data from which the customer data is going to be aggregated

        Returns:
            pd.DataFrame: a dataframe which contains the original data and the aggregated data
        """

        # group and aggregate the data
        customer_grouping = data.groupby(by="CustomerId")
        customer_aggregation = customer_grouping.agg(
            TotalTransaction = ('Amount', 'sum'),
            AverageTransaction = ('Amount', 'mean'),
            TransactionCount = ('TransactionId', 'count'),
            StdTransaction = ('Amount', 'std')
        )

        # join the newly aggregated data with the previous data over the customerId
        data = data.join(other=customer_aggregation, how='left', on='CustomerId')

        return data

    @staticmethod
    def normalize_numerical_features(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
        """
        A function that normalizes numerical data.

        **Note: Make sure to run this before categorical encoding, because normalizing categorical encodings is very wrong**

        Args:
            data(pd.DataFrame): the data whose numerical values are to be normalized
        
        Returns:
            pd.DataFrame: the dataframe with normalized numerical columns
        """

        # obtain the numerical columns
        numerical_columns = list(data._get_numeric_data().columns)

        scaler = StandardScaler()
        scaler = scaler.fit(data[numerical_columns])

        # normalized data
        data[numerical_columns] = scaler.transform(data[numerical_columns])

        return data, scaler

from typing import List
import pandas as pd


class CreditScoreEngine:
    """
    A class for organizing the functions that are concerned with generating credit score of user(s) from given transactions of the user(s).
    """

    def __init__(self, transaction_data: pd.DataFrame):
        """
        Initializes the class.

        Args:
            transaction_data(pd.DataFame): the transactional data from which credit scores are going to be determined.
        """

        self.data = transaction_data
    
    @staticmethod
    def calculate_recency(transaction_dates: pd.Series, latest_date: pd.Timestamp=None) -> int:
        """
        A function that calculates the recency of a users transaction.

        Args:
            transaction_dates(pd.Series): a series containing the dates of the user's transactions.
            latest_date(pd.Timestamp): the time from which we want to calculate the recency value. Default is the time where the function is run.

        Returns:
            int: an integer that represents the difference between the latest_date and the most recent date from the transaction date series. 
        """
        
        # If the latest date isn't passed, use today's date
        if latest_date is None:
            latest_date = pd.Timestamp.utcnow()

        # Ensure the series is in datetime format
        transaction_dates = pd.to_datetime(transaction_dates)

        # Obtain the most recent transaction date from the series
        user_latest = transaction_dates.max()

        # Calculate the time delta
        time_delta = latest_date - user_latest

        # Return the number of days in the time delta
        return time_delta.days
    
    def calcualte_rfms(self) -> pd.DataFrame:
        """
        A method that calcualtes the RFMS values of users.

        Returns:
            pd.DataFrame: a dataframe that contains the RFMS values corresponding to each user.
        """

        # convert the date string to a pd.datetime object
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        
        # first group the data for each data
        user_groupings = self.data.groupby(by="CustomerId")

        # now aggregate all the RFMS values
        rfms_data = user_groupings.agg(
            Recency=("TransactionStartTime", CreditScoreEngine.calculate_recency),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum"),
            Std_Deviation=("Value", "std")
        )

        # now fill the NA values in standard deviation with zero
        rfms_data['Std_Deviation'] = rfms_data['Std_Deviation'].fillna(value=0)

        return rfms_data
    
    def score_rfms(self, rfms_data: pd.DataFrame, rfms_weights: List[float]=[0.1, 0.2, 0.5, 0.2]) -> pd.DataFrame:
        """
        A method that converts RFMS scores into one combined score. The process is as follows:
        RFMS values will be scored 1 through 5,  5 being the highest and 1 being the lowest. The way this id done is by using quantiles as identifies of scores.
        - Recency : 5 is given to the most recent visitors and 1 is given to least recent visitors.
        - Frequency: 5 is given to users that have high frequency of transactions and 1 for those who don't
        - Monetary: 5 is given to users that have high transaction amount and 1 for those who have low transaction amounts.
        - Std_Deviation: 5 is given to users whoes transaction amounts are consistent and 1 for those that aren't.
        
        Now what is left is to combine the scores into one variable. Giving different importance to all of measures, i.e RFMS measures/scores. Here are the level of importance:
        - Recency has the lowest importance with a weight of 10% on the overall combined score
        - Monetary has the highest importance with a weight of 50% on the overall combined score
        - Frequency and Std_Deviation have the same importance with both weighing 20% on the overall combined score

        Args:
            rfms_data(pd.DataFrame): the dataframe containing RFMS values in columns of their own.
            rfms_weights(List[float]): the weights to be used for combining RFMS values to obatin a single score.(defautl 0.1, 0.2, 0.5, 0.2)

        Returns:
            pd.DataFrame: A dataframe containing the final score obtained from RFMS scores and the individual RFMS scores
        """

        # score the RFMS, 1 through 5
        recency_score = pd.cut(x=rfms_data['Recency'], bins=5, labels=list(range(5, 0, -1)))
        frequency_score = pd.cut(x=rfms_data['Frequency'], bins=5, labels=list(range(1, 6, 1)))
        monetary_score = pd.cut(x=rfms_data['Monetary'], bins=5, labels=list(range(1, 6, 1)))
        std_score = pd.cut(x=rfms_data['Std_Deviation'], bins=5, labels=list(range(5, 0, -1)))

        # combine the scored RFMS using the provided weights
        rfms_score = (recency_score.astype(int) * rfms_weights[0]) + (frequency_score.astype(int)) * rfms_weights[1] + (monetary_score.astype(int) * rfms_weights[2]) + (std_score.astype(int)) * rfms_weights[3]  

        # create a dataframe to return the results
        result = pd.DataFrame({
            "RecencyScore": recency_score.astype(int),
            "FrequencyScore": frequency_score.astype(int),
            "MonetaryScore": monetary_score.astype(int),
            "StdScore": std_score.astype(int),
            "RFMS_Score": rfms_score
        }).map(lambda x: float(x))

        return result

    def label_rfms_score(self, data: pd.DataFrame, score_column: str='RFMS_Score') -> tuple[pd.DataFrame, float]:
        """
        A function that will lable RFMS scores as being 'Good' or 'Bad'. It uses the 55th quantile as the decision boundary.

        Args:
            data(pd.DataFrame): the dataframe that contains the RFMS scores
            score_column(str): the name of the column that contains the RFMS score information

        Returns:
            pd.DataFrame: a new dataframe containing the labels of each RFMS scores in a new column called 'RiskLabel'
            float: the value of the RFMS score used as a decision boundary.
        """

        # label the scores
        values = pd.qcut(x=data[score_column], q=[0, 0.55, 1], labels=['Bad', 'Good'])

        # add the score to the dataframe
        data['RiskLabel'] = values

        # determine the boundary
        boundary = data[score_column].quantile(q=0.55)

        return data, boundary

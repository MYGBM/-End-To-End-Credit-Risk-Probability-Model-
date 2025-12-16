from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore")

class WOE_Binner:
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initialize a WOE Binner

        Args:
            data(pd.DataFrame): the data to perform WOE binning on
            target(str): a string identifying the column where the binary target is found in.
        """
        self.data = data
        self.target = target
        self.numerical_columns = self.obtain_numerical_cols()
        self.categorical_columns = self.obtain_categorical_cols()

    def obtain_numerical_cols(self) -> List[str]:
        """
        A function that returns the numerical columns from the classes dataframe, won't include the target column
        """
        numerical_columns = self.data._get_numeric_data().columns
        numerical_columns = [column for column in numerical_columns if column != self.target]

        return numerical_columns

    def obtain_categorical_cols(self) -> List[str]:
        """
        A function that returns the categorical columns from the classes dataframe, won't include the target column
        """
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target in categorical_columns:
            categorical_columns.remove(self.target)

        return categorical_columns

    def bin_numerical_cols(self, columns_to_ignore: List[str] = [], n_bins: int = 5) -> dict:
        """
        A function that will bin numeric data into a given amount of bins/groups
        """
        bins = {}
        for column in self.numerical_columns:
            if column in columns_to_ignore: continue
            col_bins = pd.qcut(x=self.data[column], q=5, duplicates='drop')
            bins[column] = col_bins

        return bins

    def bin_categorical_cols(self, ignore_cols: List[str] = []):
        """
        A function that will bin categorical data into given amount of bins/groups
        """
        # Ensure 'bin' is not part of the data being processed
        if 'bin' in self.data.columns:
            self.data = self.data.drop(columns=['bin'])
        
        bins = {}
        for column in self.categorical_columns:
            # check if the number of unique values is greater than 10
            n_unique = self.data[column].nunique()

            # if it greater than 10 skip it
            if n_unique > 10 or column in ignore_cols: continue
            
            # get the unique values for the column
            col_bins = self.data[column].unique().tolist()

            bins[column] = col_bins
        
        return bins

    def obtain_counts(self, bins: dict, good_label: any, numeric: bool = True) -> dict:
        """
        A function that will count the good and bad values for every cut

        Args:
            bins(dict): a dictionary that contains keys that are column names and values that contain the bins
            good_label(any): a value that indicates what the good value is in the target column
            numeric(bool): a bool that determines whether the bins passed are for numeric or non numeric columns
        
        Returns:
            dict: a dict that contains columns where the keys are column names and the value are dictionaries containing values for good and bad counts
        """
        counts = {}
        if numeric:
            for column, binnig in bins.items():
                self.data['bin'] = binnig

                # group the bins and count occurences of each target value
                grouped_data = self.data.groupby('bin')[self.target].value_counts().unstack(fill_value=0)

                # Store the counts in the dictionary
                counts[column] = grouped_data.to_dict(orient='index')

            # Drop the temporary 'bin' column
            self.data = self.data.drop(columns=['bin'])
        else:
            for column, binnig in bins.items():
                # group the bins and count occurences of each target value
                grouped_data = self.data.groupby(column)[self.target].value_counts().unstack(fill_value=0)

                # Store the counts in the dictionary
                counts[column] = grouped_data.to_dict(orient='index')
        return counts

    def calculate_woe(self, counts: dict) -> dict:
        """
        A function that calculates the WOE for a given counts dictionary.
    
        Args:
            counts (dict): a dict containing names of columns, within another dictionary
                           that contains keys as bins and values as dicts containing counts.
        
        Returns:
            dict: a dict that contains keys as columns and then values as dicts that
                  themselves contain floats for the WOE value.
        """
        woe_dict = {}
    
        # Calculate total good and bad across all bins
        for column, bins in counts.items():
            total_good = sum(bins[bin].get('Good', 0) for bin in bins)  # Total good counts
            total_bad = sum(bins[bin].get('Bad', 0) for bin in bins)    # Total bad counts
    
            # Store the WOE for each bin
            woe_dict[column] = {}
    
            for bin, bin_counts in bins.items():
                good_count = bin_counts.get('Good', 0)
                bad_count = bin_counts.get('Bad', 0)
    
                # Avoid division by zero by adding a small constant
                good_ratio = (good_count + 0.5) / (total_good + 0.5)
                bad_ratio = (bad_count + 0.5) / (total_bad + 0.5)
    
                # Calculate WOE
                if good_ratio > 0 and bad_ratio > 0:  # Avoid log(0)
                    woe = np.log(good_ratio / bad_ratio)
                else:
                    woe = 0
    
                woe_dict[column][bin] = woe
    
        return woe_dict
    
    def bad_probability(self, counts:dict) -> dict:
        """
        A function that calculates the bad probability given counts dictionary of bins

        Args:
            counts(dict): a dict containing names of columns and within another dictionary that contains keys as bins and values as dicts containing counts

        Returns:
            dict: a dict that contains key as columns and then values as dicts that they themselves contain floats for the bad probabilities            
        """
        bad_prob_dict = {}

        # Iterate over each column in the counts dictionary
        for column, bins in counts.items():
            bad_prob_dict[column] = {}
    
            # Calculate bad probability for each bin
            for bin, bin_counts in bins.items():
                bad_count = bin_counts.get('Bad')  # Count of bad (0) values in the current bin
                total_count = sum(bin_counts.values())  # Total count of good + bad values in the bin
    
                # Avoid division by zero
                if total_count > 0:
                    bad_prob = bad_count / total_count
                else:
                    bad_prob = 0  # If no values in the bin, bad probability is 0
    
                bad_prob_dict[column][bin] = bad_prob
    
        return bad_prob_dict
    
    @staticmethod
    def get_plotting_data(bins_dict: dict, counts: dict, bad_probs: dict, woe_dict: dict, column: str, numeric: bool):
        """
            Retrieves the necessary data for plotting based on the given column.

            Args:
                bins_dict (dict): Dictionary containing bin information for the specified column.
                counts (dict): Dictionary containing counts of good and bad loans for each bin.
                bad_probs (dict): Dictionary containing bad probability values for each bin.
                woe_dict (dict): Dictionary containing Weight of Evidence (WoE) values for each bin.
                column (str): The column name for which the data is retrieved.
                numeric (bool): Indicates if the column contains numeric or categorical values.

            Returns:
                DataFrame: A pandas DataFrame containing bins, good counts, bad counts, bad probability, and WoE values for the specified column.
        """
        if numeric == True:
            bins = bins_dict[column].unique()
        else:
            bins = bins_dict[column]

        good_counts = []
        bad_counts = []
        bad_propability = []
        woe = []

        for bin in bins:
            good_counts.append(counts[column][bin]['Good'])
            bad_counts.append(counts[column][bin]['Bad'])
            bad_propability.append(bad_probs[column][bin])
            woe.append(woe_dict[column][bin])

        ploting_data = pd.DataFrame({
            'bins': bins,
            'good_count': good_counts,
            'bad_count': bad_counts,
            'bad_probability': bad_propability,
            'woe': woe
        })

        return ploting_data

    @staticmethod
    def plot_woe_data(plotting_data: pd.DataFrame, column_name: str):
        """
        Plots the distribution of Good and Bad loan counts, along with WoE and Bad Probability, for a given set of RFMS bins.

        Args:
            plotting_data (pd.DataFrame): A DataFrame containing the data for plotting, with the following columns:
                - 'bins': The RFMS score bins.
                - 'good_count': Count of good loans in each bin.
                - 'bad_count': Count of bad loans in each bin.
                - 'bad_probability': The bad probability for each bin.
                - 'woe': Weight of Evidence (WoE) values for each bin.
            column_name (str): the name whoes woe data is being plotted

        Returns:
            A plot showing the count distribution of Good and Bad loans (as bar plots), WoE (as a blue line), and Bad Probability (as a red line).
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        bar_width = 0.4
        index = plotting_data['bins'].index.astype(int)

        # Create bar plots for Good and Bad counts
        bar1 = ax1.bar(index - bar_width/2, plotting_data['good_count'], width=bar_width, label='Good', color='lightgreen')
        bar2 = ax1.bar(index + bar_width/2, plotting_data['bad_count'], width=bar_width, label='Bad', color='lightcoral')

        ax1.set_xlabel(f'{column_name} Bins', weight='bold', labelpad=20)
        ax1.set_ylabel('Count Distribution', weight='bold', labelpad=20)
        ax1.set_title(f'Distribution of Good and Bad Loans by {column_name} Bins', weight='bold', fontsize=20, pad=20)
        ax1.legend(loc='upper left')

        # Adding WoE line
        ax2 = ax1.twinx()
        ax2.plot(index, plotting_data['woe'], color='blue', marker='o', label='WoE', linewidth=2)
        ax2.set_ylabel('WoE', color='blue')

        # Annotate WoE values on the line
        for i, woe in enumerate(plotting_data['woe']):
            ax2.text(index[i]+0.1, woe, f'{woe:.2f}', color='blue', ha='center', fontsize=10)

        # Adding Bad Probability
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outwards
        ax3.plot(index, plotting_data['bad_probability'], color='red', marker='s', label='Bad Probability', linewidth=2)
        ax3.set_ylabel('Bad Probability', color='red')

        # Annotate Bad Probability values on the line
        for i, prob in enumerate(plotting_data['bad_probability']):
            ax3.text(index[i]+0.1, prob, f'{prob:.2f}', color='red', ha='center', fontsize=10)

        # Set custom x-tick labels using the 'bins' column
        plt.xticks(index, plotting_data['bins'], rotation=45, ha='right')  # Rotate for better readability

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiple_woe_data(plotting_data_list: list, column_names: list, n_cols: int):
        """
        Plots the distribution of Good and Bad loan counts, along with WoE and Bad Probability, for multiple columns in a grid layout.

        Args:
            plotting_data_list (list): A list of DataFrames containing the data for plotting.
            column_names (list): A list of column names corresponding to the data.
            n_cols (int): Number of columns in the subplot grid.

        Returns:
            A plot with subplots showing the count distribution of Good and Bad loans (as bar plots), WoE (as a blue line), and Bad Probability (as a red line) for multiple columns.
        """
        n_rows = int(np.ceil(len(column_names) / n_cols))  # Calculate the number of rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))
        axes = axes.flatten()  # Flatten the axes array to easily iterate over it

        for i, (plotting_data, column_name) in enumerate(zip(plotting_data_list, column_names)):
            ax1 = axes[i]

            bar_width = 0.4
            index = np.arange(len(plotting_data['bins']))  # Use the range for index

            # Create bar plots for Good and Bad counts
            bar1 = ax1.bar(index - bar_width/2, plotting_data['good_count'], width=bar_width, label='Good', color='lightgreen')
            bar2 = ax1.bar(index + bar_width/2, plotting_data['bad_count'], width=bar_width, label='Bad', color='lightcoral')

            ax1.set_xlabel(f'{column_name} Bins')
            ax1.set_ylabel('Count Distribution')
            ax1.set_title(f'{column_name}', weight='bold', fontsize=14)
            ax1.legend(loc='upper right')

            # Adding WoE line
            ax2 = ax1.twinx()
            ax2.plot(index, plotting_data['woe'], color='blue', marker='o', label='WoE', linewidth=2)
            ax2.set_ylabel('WoE', color='blue')

            # Annotate WoE values on the line
            for j, woe in enumerate(plotting_data['woe']):
                ax2.text(index[j], woe, f'{woe:.2f}', color='blue', ha='center', fontsize=8)

            # Adding Bad Probability
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outwards
            ax3.plot(index, plotting_data['bad_probability'], color='red', marker='s', label='Bad Probability', linewidth=2)
            ax3.set_ylabel('Bad Probability', color='red')

            # Annotate Bad Probability values on the line
            for j, prob in enumerate(plotting_data['bad_probability']):
                ax3.text(index[j], prob, f'{prob:.2f}', color='red', ha='center', fontsize=8)

            # Set custom x-tick labels using the 'bins' column
            ax1.set_xticks(index)
            ax1.set_xticklabels(plotting_data['bins'], rotation=45, ha='right')

        # Remove any unused subplots
        for i in range(len(column_names), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def calculate_iv_from_bins(counts: dict, woe_values: dict) -> dict:
        """
        Calculate Information Value (IV) for multiple columns based on their bins, good/bad counts, and WoE values.

        Args:
            counts (dict): Dictionary of good/bad counts for each bin of each column. 
                           Structure: {column: {bin: {'Good': value, 'Bad': value}}}
            woe_values (dict): Dictionary of WoE values for each bin of each column.
                               Structure: {column: {bin: woe_value}}

        Returns:
            iv_values (dict): Dictionary with IV values for each column.
        """
        iv_values = {}

        # Loop through each column
        for column, bins in counts.items():
            iv = 0  # Initialize IV for the column

            # Get total Good and Bad counts for the column
            total_good = sum(bin_counts['Good'] for bin_counts in bins.values())
            total_bad = sum(bin_counts['Bad'] for bin_counts in bins.values())

            # Loop through each bin in the column
            for bin_label, bin_counts in bins.items():
                good = bin_counts['Good']
                bad = bin_counts['Bad']

                # Calculate percentages of good and bad for the bin
                good_perc = good / total_good if total_good > 0 else 0
                bad_perc = bad / total_bad if total_bad > 0 else 0

                # Get WoE for the bin
                woe = woe_values[column].get(bin_label, 0)

                # Calculate IV contribution for the bin
                iv += (good_perc - bad_perc) * woe

            # Store IV for the column
            iv_values[column] = iv

        return iv_values
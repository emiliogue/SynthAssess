import pandas as pd
class SimilarityMetrics:
    def __init__(self, original_data, synthetic_data):
        self.original_data = original_data
        self.synthetic_data = synthetic_data

    def calculate_range_coverage(self, categorical_columns=[]):
        '''
        Calculate the range coverage of the synthetic data compared to the original data.
        Range coverage is the percentage of the original data range that is covered by the synthetic data.
        For categorical columns, range coverage is the percentage of original values that are present in the synthetic data.
        
        param: categorical_columns: list of categorical columns in the dataset
        return: pandas dataframe with range coverage for each column
        '''
        # Calculate how much of the original data range is covered by the synthetic data
        range_coverage = {}
        for column in self.original_data.columns:
            if column in categorical_columns:
                original_values = set(self.original_data[column].unique())
                synthetic_values = set(self.synthetic_data[column].unique())
                coverage = len(synthetic_values.intersection(original_values)) / len(original_values)
                range_coverage[column] = coverage
            else:
                original_min = self.original_data[column].min()
                original_max = self.original_data[column].max()
                synthetic_min = self.synthetic_data[column].min()
                synthetic_max = self.synthetic_data[column].max()
                coverage = (min(synthetic_max, original_max) - max(synthetic_min, original_min)) / (original_max - original_min)
                range_coverage[column] = coverage
        # convert to percentage and round to 2 decimal places
        range_coverage = {k: round(v * 100, 2) for k, v in range_coverage.items()}

        # convert to pandas dataframe for easier visualization including column names and sorting
        range_coverage = pd.DataFrame.from_dict(range_coverage, orient='index', columns=['Range Coverage (%)']).sort_values(by='Range Coverage (%)', ascending=False)

        # add mean range coverage
        mean_range_coverage = range_coverage['Range Coverage (%)'].mean()
        range_coverage.loc['Mean Range Coverage'] = mean_range_coverage
        range_coverage = range_coverage.reset_index().rename(columns={'index': 'Column'})

        return range_coverage
    

    def calculate_correlation_matrix(self):
        # Calculate correlation matrix for both original and synthetic data
        original_corr = self.original_data.corr()
        synthetic_corr = self.synthetic_data.corr()

        return original_corr, synthetic_corr
    
    def kolmogorov_smirnov_test(self):
        '''
        Kolmogorov-Smirnov test for similarity between original and synthetic data.
        return: ks_statistic, p_value
        '''

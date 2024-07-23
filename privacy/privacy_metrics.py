import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from SynthAssess.utils.data_preprocessing import precalculate_distances
from matplotlib import pyplot as plt
import seaborn as sns
from anonymeter.evaluators import SinglingOutEvaluator
from anonymeter.evaluators import LinkabilityEvaluator
from anonymeter.evaluators import InferenceEvaluator
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

class PrivacyMetrics:
    def __init__(self, original_data, synthetic_data, holdout_data=None):
        '''
        Initialize the PrivacyMetrics class.
        :param original_data: Original dataset.
        :param synthetic_data: Synthetic dataset.
        :param holdout_data: Holdout dataset(default is None).
        '''
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.original_distances = None
        self.holdout_data = holdout_data
        self.benchmark_distances = None
        self.descriptive_stats_benchmark = None
        self.synthetic_distances = None
        self.descriptive_stats_synthetic = None
        self.nneighbours_benchmark = None
        self.nneighbours_synthetic = None
        self.nneighbours_threshold = None
        # Define distances
        self.holdout_x_synthetic_distances = self.compute_average_knn_distance_between_samples(anchor=self.holdout_data, comparison=self.synthetic_data)[0]
        self.holdout_x_original_distances = self.compute_average_knn_distance_between_samples(anchor=self.holdout_data, comparison=self.original_data)[0]
        self.synthetic_x_original_distances = self.compute_average_knn_distance_between_samples(anchor=self.synthetic_data, comparison=self.original_data)[0]

    def execute_singling_out_attack(self):
        '''
        Execute singling out attack on the synthetic data. Attack is performed by measuring the risk of singling out a specific individual in the dataset.
        :return: Results of the privacy attacks.
        '''
        evaluator = SinglingOutEvaluator(ori=self.original_data , syn=self.synthetic_data, control=self.holdout_data, n_attacks= 50)
        evaluator.evaluate()
        risk = evaluator.risk(confidence_level=0.95)
        res = evaluator.results()

        return risk, res.attack_rate, res.baseline_rate, res.control_rate
    
    def execute_linkability_attack(self, aux_columns=[]):
        '''
        Execute linkability attack on the synthetic data.
        :param aux_columns: List of two sets of columns to be linked. Attack is performed by measuring the risk of linking two datasets containing the specified columns.
        :return: Results of the privacy attacks.
        '''
        if aux_columns == []:
            print("No auxiliary columns provided. Please define auxiliary columns.")
            aux_columns = [['education', 'hours-per-week', 'capital-loss', 'capital-gain'], [ 'race', 'sex', 'fnlwgt', 'age', 'native-country']]
            
        evaluator = LinkabilityEvaluator(ori=self.original_data , syn=self.synthetic_data, control=self.holdout_data, n_attacks= 50, aux_cols=aux_columns, n_neighbors=10)
        evaluator.evaluate(n_jobs=-1)
        risk = evaluator.risk(confidence_level=0.95)
        res = evaluator.results()

        return risk, res.attack_rate, res.baseline_rate, res.control_rate

    def execute_inference_attack(self):
        '''
        Execute inference attack on the synthetic data. Attack is performed by measuring the risk of inferring secret columns given access to all other columns.
        :return: Plotly figure showing the measured inference risk for secret columns.
        '''
        columns = self.original_data.columns
        results = []
        
        for secret in columns:
            aux_columns = [col for col in columns if col != secret]
            evaluator = InferenceEvaluator(ori=self.original_data, syn=self.synthetic_data, control=self.holdout_data, aux_cols=aux_columns, secret=secret, n_attacks=1000)
            evaluator.evaluate(n_jobs=-1)
            results.append((secret, evaluator.results()))
        
        risks = [res[1].risk().value for res in results]
        columns = [res[0] for res in results]

        fig = go.Figure(data=[
            go.Bar(
                x=columns,
                y=risks,
                marker=dict(color='rgba(255, 153, 51, 0.5)', line=dict(color='black', width=1.5))
            )
        ])
        fig.update_layout(
            title='Measured Inference Risk for Secret Columns',
            xaxis_title='Secret Column',
            yaxis_title='Measured Inference Risk',
            xaxis_tickangle=-45
        )
        return fig
    
    def compute_knn_distance_benchmark(self, k=5, s=10, metric='euclidean'):
        """
        Calculate the average k-NN distance between rows in the original data and a 50% random sample.
        :param k: Number of nearest neighbors.
        :param s: Number of random samples to eliminate bias.
        :return: Descriptive statistics of the k-NN distance distribution.
        """
        self.original_distances = precalculate_distances(self.original_data, metric=metric)
        distances = []
        n_samples = len(self.original_data)
        sample_size = n_samples // 2
        
        for _ in range(s):
            # Create a random 50% sample from the original data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_distances = self.original_distances[sample_indices, :]
            if sample_distances.shape[0] < k:
                raise ValueError("Sample size too small for k neighbors")
            # Compute distances for each row in the original data
            for i in range(n_samples):
                row_distances = sample_distances[:, i]
                knn_distances = np.partition(row_distances, k)[:k]
                distances.append(np.mean(knn_distances))
        
        # Convert distances to a pandas Series for easier descriptive statistics calculation
        distances_series = pd.Series(distances)
        self.benchmark_distances = distances_series
        
        # Calculate descriptive statistics
        descriptive_stats = distances_series.describe(percentiles=[.25, .5, .75]).to_dict()
        
        return descriptive_stats
        ## potential additional function to bucket the rows into categories and/or plot the distribution
    
    def compute_knn_distance_original_synthetic(self, k=5, s=10, metric='euclidean'):
        """
        Calculate the average distance between each row in the synthetic/holdout data and its k-NN in the original data. Number of sampling variations can be specified.
        :param k: Number of nearest neighbors.
        :param s: Number of random samples to eliminate bias.
        :return: Descriptive statistics of the k-NN distance distribution.
        """
        combined_data = pd.concat([self.original_data, self.synthetic_data])
        self.synthetic_distances = precalculate_distances(combined_data, metric=metric)
        self.synthetic_only = self.synthetic_distances[len(self.original_data):, len(self.original_data):]
        distances = []
        n_samples = len(self.synthetic_data)
        sample_size = n_samples // 2
        
        for _ in range(s):
            # Create a random 50% sample from the original data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_distances = self.synthetic_only[sample_indices, :]
            if sample_distances.shape[0] < k:
                raise ValueError("Sample size too small for k neighbors")
            # Compute distances for each row in the original data
            for i in range(n_samples):
                row_distances = sample_distances[:, i]
                knn_distances = np.partition(row_distances, k)[:k]
                distances.append(np.mean(knn_distances))
        
        # Convert distances to a pandas Series for easier descriptive statistics calculation
        distances_series = pd.Series(distances)
        self.synthetic_distances = distances_series
        
        # Calculate descriptive statistics
        descriptive_stats = distances_series.describe(percentiles=[.25, .5, .75]).to_dict()
        
        return descriptive_stats
        ## potential additional function to bucket the rows into categories and/or plot the distribution

    def compute_average_knn_distance_between_samples(self, k=5, s=10, metric='euclidean', anchor=None, comparison=None):
        """
        Calculate the average distance between each row in the comparison data and its k-NN in the anchor data. Number of sampling variations can be specified.
        :param k: Number of nearest neighbors.
        :param s: Number of random samples to eliminate bias.
        :param anchor: Anchor data to compare against (default is the original data).
        :param comparison: Comparison data to calculate distances for (default is the synthetic data).
        :return: Descriptive statistics of the k-NN distance distribution.
        """
        if anchor is None:
            anchor = self.original_data
        if comparison is None:
            comparison = self.synthetic_data
            
        combined_data = pd.concat([anchor, comparison])
        combined_distances = precalculate_distances(combined_data, metric=metric)
        anchor_x_compared_distances = combined_distances[len(anchor):, len(anchor):]
        distances = []
        n_samples = len(comparison)
        sample_size = n_samples // 2
        
        for _ in range(s):
            # Create a random 50% sample from the original data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_distances = anchor_x_compared_distances[sample_indices, :]
            if sample_distances.shape[0] < k:
                raise ValueError("Sample size too small for k neighbors")
            # Compute distances for each row in the original data
            for i in range(n_samples):
                row_distances = sample_distances[:, i]
                knn_distances = np.partition(row_distances, k)[:k]
                distances.append(np.mean(knn_distances))
        
        # Convert distances to a pandas Series for easier descriptive statistics calculation
        distances_series = pd.Series(distances)
        
        # Calculate descriptive statistics
        descriptive_stats = distances_series.describe(percentiles=[.25, .5, .75]).to_dict()
        
        return distances_series, descriptive_stats
    
    def compute_nneighbors_benchmark(self, threshold=0.5, s=10, metric='euclidean'):
        """
        Calculate the number of nearest neighbors within a threshold distance for each row in the synthetic data.
        :param threshold: Distance threshold.
        :param s: Number of random samples to eliminate bias.
        :param metric: Distance metric (default is 'euclidean').
        :return: Descriptive statistics of the number of neighbors within the threshold.
        """

        self.nneighbours_threshold = threshold
        self.original_distances = precalculate_distances(self.original_data, metric=metric)
        counts = []
        n_samples = len(self.original_data)
        sample_size = n_samples // 2

        for _ in range(s):
            # Create a random 50% sample from the original data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_distances = self.original_distances[sample_indices, :]
            
            # Compute the count of neighbors within the threshold distance for each row in the original data
            for i in range(n_samples):
                row_distances = sample_distances[:, i]
                count_within_threshold = np.sum(row_distances <= threshold)
                counts.append(count_within_threshold)
        
        # Convert distances to a pandas Series for easier descriptive statistics calculation
        n_neighbors_series = pd.Series(counts)
        self.nneighbours_benchmark = n_neighbors_series
        
        # Calculate descriptive statistics
        descriptive_stats = n_neighbors_series.describe(percentiles=[.25, .5, .75]).to_dict()
        
        
        return descriptive_stats
        ## potential additional function to bucket the counts into categories and/or plot the distribution

    def compute_average_knn_distance_return_distances(self, k=5, s=10, metric='euclidean', anchor=None, comparison=None):
        """
        Calculate the average distance between each row in the comparison data and its k-NN in the anchor data.
        Number of sampling variations can be specified.
        :param k: Number of nearest neighbors.
        :param s: Number of random samples to eliminate bias.
        :param anchor: Anchor data to compare against (default is the original data).
        :param comparison: Comparison data to calculate distances for (default is the synthetic data).
        :return: Descriptive statistics of the k-NN distance distribution.
        """
        if anchor is None:
            anchor = self.original_data
        if comparison is None:
            comparison = self.synthetic_data

        combined_data = pd.concat([anchor, comparison])
        combined_distances = precalculate_distances(combined_data, metric=metric)
        anchor_x_compared_distances = combined_distances[len(anchor):, :len(anchor)]
        distances = []
        n_samples = len(comparison)
        sample_size = n_samples // 2

        all_knn_distances = []

        for _ in range(s):
            # Create a random 50% sample from the comparison data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_distances = anchor_x_compared_distances[sample_indices, :]
            if sample_distances.shape[1] < k:
                raise ValueError("Sample size too small for k neighbors")
            # Compute distances for each row in the comparison data
            for i in range(sample_size):
                row_distances = sample_distances[i, :]
                knn_distances = np.partition(row_distances, k)[:k]
                distances.append(np.mean(knn_distances))
                all_knn_distances.append(knn_distances)

        all_knn_distances = np.array(all_knn_distances)
        
        return all_knn_distances

    # def generate_probability_matrix(self, knn_distances, num_buckets=10):
    #     def calculate_probability_matrix(knn_distances, neighbor_buckets, distance_buckets, num_buckets):
    #         probability_matrix = np.zeros((num_buckets, num_buckets))

    #         for i in range(num_buckets):
    #             for j in range(num_buckets):
    #                 k = int(neighbor_buckets[i + 1])
    #                 dist = distance_buckets[j + 1]
    #                 count_within_threshold = np.sum(knn_distances[:, :min(k, knn_distances.shape[1])] <= dist)
    #                 total_counts = knn_distances[:, :min(k, knn_distances.shape[1])].size
    #                 probability_matrix[i, j] = count_within_threshold / total_counts

    #         return probability_matrix

    #     max_samples = knn_distances.shape[0]
    #     max_neighbors = min(knn_distances.shape[1], max_samples - 1)
    #     max_distance = np.max(knn_distances)

    #     # Initial logarithmic bucket ranges
    #     neighbor_buckets = np.logspace(0, np.log10(max_neighbors + 1), num=num_buckets + 1)
    #     distance_buckets = np.logspace(0, np.log10(max_distance), num=num_buckets + 1)

    #     prev_neighbor_buckets = neighbor_buckets.copy()
    #     prev_distance_buckets = distance_buckets.copy()

    #     while True:
    #         probability_matrix = calculate_probability_matrix(knn_distances, neighbor_buckets, distance_buckets, num_buckets)

    #         if not (np.any(np.all(probability_matrix == 1, axis=0)) or np.any(np.all(probability_matrix == 1, axis=1))):
    #             break

    #         # Adjust neighbor_buckets and distance_buckets if necessary
    #         if np.any(np.all(probability_matrix == 1, axis=0)):
    #             col_index = np.argmax(np.all(probability_matrix == 1, axis=0))
    #             max_distance = np.max(knn_distances[:, :int(neighbor_buckets[col_index + 1])])
    #             distance_buckets = np.logspace(0, np.log10(max_distance), num=num_buckets + 1)

    #         if np.any(np.all(probability_matrix == 1, axis=1)):
    #             row_index = np.argmax(np.all(probability_matrix == 1, axis=1))
    #             max_neighbors = int(neighbor_buckets[row_index + 1])
    #             neighbor_buckets = np.logspace(0, np.log10(max_neighbors + 1), num=num_buckets + 1)

    #         # Check for termination condition to avoid infinite loop
    #         if np.array_equal(prev_neighbor_buckets, neighbor_buckets) and np.array_equal(prev_distance_buckets, distance_buckets):
    #             break

    #         prev_neighbor_buckets = neighbor_buckets.copy()
    #         prev_distance_buckets = distance_buckets.copy()

    #     # Plotting the probability matrix
    #     plt.figure(figsize=(12, 8))
    #     ax = sns.heatmap(probability_matrix, annot=True, fmt=".2f", cmap='YlGnBu', 
    #                     xticklabels=np.round(distance_buckets[1:], 2), 
    #                     yticklabels=np.round(neighbor_buckets[1:], 2))
    #     ax.set_xlabel('Distance Buckets')
    #     ax.set_ylabel('Number of Neighbors Buckets')
    #     ax.set_title('Probability Matrix Heatmap')
    #     plt.show()

    #     fig = go.Figure(data=go.Heatmap(
    #         z=probability_matrix,
    #         x=np.round(distance_buckets[1:], 2),
    #         y=np.round(neighbor_buckets[1:], 2),
    #         colorscale='YlGnBu',
    #         zmin=0,  # Optional: specify the min value for the color scale
    #         zmax=1,  # Optional: specify the max value for the color scale
    #         colorbar=dict(title='Probability')
    #     ))

    #     # Focus on the area of importance by limiting the range of the axes
    #     fig.update_layout(
    #         title='Probability Matrix Heatmap',
    #         xaxis_title='Distance Buckets',
    #         yaxis_title='Number of Neighbors Buckets',
    #         xaxis=dict(range=[min(np.round(distance_buckets[1:], 2)), max(np.round(distance_buckets[1:], 2))]),
    #         yaxis=dict(range=[min(np.round(neighbor_buckets[1:], 2)), max(np.round(neighbor_buckets[1:], 2))]),
    #     )

    #     fig.show()

    #     # Calculate the difference score
    #     ideal_matrix = np.ones((num_buckets, num_buckets))
    #     difference_matrix = ideal_matrix - probability_matrix
    #     difference_score = np.linalg.norm(difference_matrix) / np.linalg.norm(ideal_matrix) * 100

    #     return ax, difference_score
    
    def generate_probability_matrix(self, knn_distances, num_buckets=10):
        '''
        Generate a probability matrix based on the k-NN distances.
        :param knn_distances: k-NN distances.'''
        def calculate_probability_matrix(knn_distances, neighbor_buckets, distance_buckets, num_buckets):
            probability_matrix = np.zeros((num_buckets, num_buckets))

            for i in range(num_buckets):
                for j in range(num_buckets):
                    k = int(neighbor_buckets[i + 1])
                    dist = distance_buckets[j + 1]
                    count_within_threshold = np.sum(knn_distances[:, :min(k, knn_distances.shape[1])] <= dist)
                    total_counts = knn_distances[:, :min(k, knn_distances.shape[1])].size
                    probability_matrix[i, j] = count_within_threshold / total_counts

            return probability_matrix

        max_samples = knn_distances.shape[0]
        max_neighbors = min(knn_distances.shape[1], max_samples - 1)
        max_distance = np.max(knn_distances)

        # Initial logarithmic bucket ranges
        neighbor_buckets = np.logspace(0, np.log10(max_neighbors + 1), num=num_buckets + 1)
        distance_buckets = np.logspace(0, np.log10(max_distance), num=num_buckets + 1)

        prev_neighbor_buckets = neighbor_buckets.copy()
        prev_distance_buckets = distance_buckets.copy()

        while True:
            probability_matrix = calculate_probability_matrix(knn_distances, neighbor_buckets, distance_buckets, num_buckets)

            if not (np.any(np.all(probability_matrix == 1, axis=0)) or np.any(np.all(probability_matrix == 1, axis=1))):
                break

            # Adjust neighbor_buckets and distance_buckets if necessary
            if np.any(np.all(probability_matrix == 1, axis=0)):
                col_index = np.argmax(np.all(probability_matrix == 1, axis=0))
                max_distance = np.max(knn_distances[:, :int(neighbor_buckets[col_index + 1])])
                distance_buckets = np.logspace(0, np.log10(max_distance), num=num_buckets + 1)

            if np.any(np.all(probability_matrix == 1, axis=1)):
                row_index = np.argmax(np.all(probability_matrix == 1, axis=1))
                max_neighbors = int(neighbor_buckets[row_index + 1])
                neighbor_buckets = np.logspace(0, np.log10(max_neighbors + 1), num=num_buckets + 1)

            # Check for termination condition to avoid infinite loop
            if np.array_equal(prev_neighbor_buckets, neighbor_buckets) and np.array_equal(prev_distance_buckets, distance_buckets):
                break

            prev_neighbor_buckets = neighbor_buckets.copy()
            prev_distance_buckets = distance_buckets.copy()

        # Plotting the probability matrix using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=probability_matrix,
            x=np.round(distance_buckets[1:], 2),
            y=np.round(neighbor_buckets[1:], 2),
            colorscale='YlGnBu',
            zmin=0,  # Optional: specify the min value for the color scale
            zmax=1,  # Optional: specify the max value for the color scale
            colorbar=dict(title='Probability')
        ))

        # Focus on the area of importance by limiting the range of the axes
        initial_x_range = [min(np.round(distance_buckets[1:], 2)), max(np.round(distance_buckets[1:], 2))]
        initial_y_range = [min(np.round(neighbor_buckets[1:], 2)), max(np.round(neighbor_buckets[1:], 2))]
        fig.update_layout(
            title='Probability Matrix Heatmap',
            xaxis_title='Distance Buckets',
            yaxis_title='Number of Neighbors Buckets',
            xaxis=dict(range=initial_x_range),
            yaxis=dict(range=initial_y_range),
        )

        fig.show()

        # Calculate the difference score
        ideal_matrix = np.ones((num_buckets, num_buckets))
        difference_matrix = ideal_matrix - probability_matrix
        difference_score = np.linalg.norm(difference_matrix) / np.linalg.norm(ideal_matrix) * 100

        return fig, difference_score
    def k_anonymity(self, quasi_identifiers):
        """
        Calculate k-anonymity. 
        :param quasi_identifiers: List of columns considered quasi-identifiers.
        :return: Minimum k-anonymity value.
        """
        original_groups = self.original_data.groupby(quasi_identifiers).size()
        synthetic_groups = self.synthetic_data.groupby(quasi_identifiers).size()
        
        min_k_original = original_groups.min()
        min_k_synthetic = synthetic_groups.min()
        
        return min(min_k_original, min_k_synthetic)

    def l_diversity(self, sensitive_attribute, quasi_identifiers):
        """
        Calculate l-diversity.
        :param sensitive_attribute: Column considered as the sensitive attribute.
        :param quasi_identifiers: List of columns considered quasi-identifiers.
        :return: Minimum l-diversity value.
        """
        def calculate_l_diversity(df):
            groups = df.groupby(quasi_identifiers)
            diversities = []
            for _, group in groups:
                diversity = group[sensitive_attribute].nunique()
                diversities.append(diversity)
            return min(diversities)

        l_diversity_original = calculate_l_diversity(self.original_data)
        l_diversity_synthetic = calculate_l_diversity(self.synthetic_data)
        
        return min(l_diversity_original, l_diversity_synthetic)

    def t_closeness(self, sensitive_attribute, quasi_identifiers):
        """
        Calculate t-closeness.
        :param sensitive_attribute: Column considered as the sensitive attribute.
        :param quasi_identifiers: List of columns considered quasi-identifiers.
        :return: Maximum t-closeness value.
        """
        def calculate_t_closeness(df, overall_distribution):
            groups = df.groupby(quasi_identifiers)
            max_t_closeness = 0
            for _, group in groups:
                group_distribution = group[sensitive_attribute].value_counts(normalize=True)
                kl_divergence = entropy(group_distribution, overall_distribution)
                max_t_closeness = max(max_t_closeness, kl_divergence)
            return max_t_closeness

        overall_distribution_original = self.original_data[sensitive_attribute].value_counts(normalize=True)
        overall_distribution_synthetic = self.synthetic_data[sensitive_attribute].value_counts(normalize=True)
        
        t_closeness_original = calculate_t_closeness(self.original_data, overall_distribution_original)
        t_closeness_synthetic = calculate_t_closeness(self.synthetic_data, overall_distribution_synthetic)
        
        return max(t_closeness_original, t_closeness_synthetic)

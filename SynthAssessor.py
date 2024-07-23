from SynthAssess.privacy.privacy_metrics import PrivacyMetrics
from SynthAssess.similarity.similarity_metrics import SimilarityMetrics
from SynthAssess.visualisation.visualisation import Visualisation
from SynthAssess.report.report_generator import ReportGenerator
from SynthAssess.ml_efficacy.ml_efficacy_metrics import MLEfficacyMetrics
from SynthAssess.utils.data_validation import verify_input_parameters
from SynthAssess.utils.data_preprocessing import encode_dataframes

class SynthAssessor:
    def __init__(self, original_data, synthetic_data, holdout_data=None, target_type='classification', target_column=None, categorical_columns=None, ordinal_columns=None):
        """
        Initialise the SynthAssessor object.
        :param original_data: The original data.
        :param synthetic_data: The synthetic data.
        :param holdout_data: The holdout data.
        :param target_type: The type of the target variable ('classification' or 'regression').
        :param target_column: The target column name.
        :param categorical_columns: List of categorical columns.
        :param ordinal_columns: List of ordinal columns.
        """
        self.original_data_raw = original_data.copy()
        self.synthetic_data_raw = synthetic_data.copy()
        self.holdout_data_raw = holdout_data.copy() if holdout_data is not None else None
        self.target_type = target_type
        self.target_column = target_column
        verify_input_parameters(self.original_data_raw, self.synthetic_data_raw, self.holdout_data_raw)
        
        if categorical_columns or ordinal_columns is not None:
            self.categorical_columns = categorical_columns
            self.ordinal_columns = ordinal_columns
            self.original_data, self.synthetic_data, self.holdout_data, self.encoder = encode_dataframes(self.original_data_raw, self.synthetic_data_raw, self.holdout_data_raw, self.categorical_columns, self.ordinal_columns)
        else:
            self.original_data = self.original_data_raw
            self.synthetic_data = self.synthetic_data_raw
            self.holdout_data = self.holdout_data_raw
            self.encoder = None

        self.privacy_metrics = PrivacyMetrics(original_data, synthetic_data, holdout_data)
        self.visualisation = Visualisation(original_data, synthetic_data, holdout_data)
        self.similarity_metrics = SimilarityMetrics(original_data, synthetic_data)
        if target_column is not None:
            self.ml_efficacy = MLEfficacyMetrics(original_data, synthetic_data, holdout_data, target_type=target_type, target_column=target_column)
        self.report = ReportGenerator()

    def generate_report(self, report_name='report.html', report_title='SynthAssess Report', privacy=True, ml_efficacy=True, similarity=True):
        """
        Generate a report for the synthetic data assessment.
        :param report_name: The name of the report file.
        """
        
        self.report.report_title = report_title

        print('Generating report...')

        #### Data Overview Section ####
        self.report.add_dataframe_to_section('Data Overview', self.original_data.sample(5), subtitle='Original Data Sample')
        self.report.add_dataframe_to_section('Data Overview', self.synthetic_data.sample(5), subtitle='Synthetic Data Sample')
        self.report.add_dataframe_to_section('Data Overview', self.similarity_metrics.calculate_range_coverage(), subtitle='Range Coverage')

        #### Statistical Similarity Section ####
        if similarity:
            ### Descriptive Statistics ###
            self.report.add_subsection('Statistical Similarity', 'Descriptive Statistics')
            self.report.add_dataframe_to_section('Statistical Similarity', self.original_data.describe().round(2).reset_index(), subtitle='Descriptive Statistics for Original Data', subsection='Descriptive Statistics')
            self.report.add_dataframe_to_section('Statistical Similarity', self.synthetic_data.describe().round(2).reset_index(), subtitle='Descriptive Statistics for Synthetic Data', subsection='Descriptive Statistics')
            self.report.add_figure_to_section('Statistical Similarity', self.visualisation.plot_descriptive_statistics(), subtitle='Comparison of Descriptive Statistics', is_plotly=True, subsection='Descriptive Statistics')
            ### Correlation ###
            correlation_array, average_correlation_diff = self.visualisation.plot_correlation_matrix_array()
            self.report.add_subsection('Statistical Similarity', f'Bivariate Correlation - Average difference: {average_correlation_diff}')
            self.report.add_figure_to_section('Statistical Similarity', correlation_array, subtitle='Bivariate Correlation Matrix', is_plotly=True, subsection=f'Bivariate Correlation - Average difference: {average_correlation_diff}')
            ### Distribution Comparison ###
            self.report.add_subsection('Statistical Similarity', 'Distribution Comparison')
            self.report.add_figure_to_section('Statistical Similarity', self.visualisation.create_plotly_scatter_comparison_with_dropdowns(), subtitle='Scatter Plot Comparison', is_plotly=True, subsection='Distribution Comparison')
            self.report.add_figure_to_section('Statistical Similarity', self.visualisation.plot_descriptive_statistics_radar(), subtitle='Descriptive Statistics as Radar Chart', is_plotly=True, subsection='Distribution Comparison')
        else:
            self.report.add_text_to_section('Statistical Similarity', 'Similarity metrics will not be calculated.')

        #### Privacy Section ####
        if privacy:
            ### k-NN Distance Measures ###
            self.report.add_subsection('Privacy', 'k-NN Distance Measures *prototype metric*')
            benchmark_text = f"Average k-NN Distance for Original Samples: {self.privacy_metrics.compute_knn_distance_benchmark()}"
            synthetic_text = f"Average k-NN Distance for Synthetic Samples: {self.privacy_metrics.compute_knn_distance_original_synthetic()}"
            benchmark_neighbours_text = f"Average Neighbours for Original Samples: {self.privacy_metrics.compute_nneighbors_benchmark()}"
            self.report.add_text_to_section('Privacy', benchmark_text, subsection='k-NN Distance Measures *prototype metric*')
            self.report.add_text_to_section('Privacy', synthetic_text, subsection='k-NN Distance Measures *prototype metric*')
            self.report.add_text_to_section('Privacy', benchmark_neighbours_text, subsection='k-NN Distance Measures *prototype metric*')
            self.report.add_figure_to_section('Privacy', self.visualisation.plot_knn_distance_benchmarks(self.privacy_metrics), subtitle='k-NN Distance Benchmark', is_plotly=True, subsection='k-NN Distance Measures *prototype metric*')
            self.report.add_figure_to_section('Privacy', self.visualisation.plot_nneighbours_benchmarks(self.privacy_metrics), subtitle='NNeighbours for Original Sample', is_plotly=True, subsection='k-NN Distance Measures *prototype metric*')
            self.distances = self.privacy_metrics.compute_average_knn_distance_return_distances()
            self.distance_matrix, self.distance_score = self.privacy_metrics.generate_probability_matrix(knn_distances=self.distances)
            self.report.add_text_to_section('Privacy', f"Privacy Matrix Difference: {self.distance_score}", subsection='k-NN Distance Measures *prototype metric*')
            self.report.add_figure_to_section('Privacy', self.distance_matrix, subtitle='Privacy Matrix', subsection='k-NN Distance Measures *prototype metric*', is_plotly=True)

            ### Privacy Attacks ###
            self.report.add_subsection('Privacy', 'Privacy Attacks')
            self.report.add_text_to_section('Privacy', "the main privacy attack, in which the attacker uses the synthetic data to guess information on records in the original data.", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', "the baseline attack, which models a naive attacker who ignores the synthetic data and guess randomly.", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', "the control privacy attack, in which the attacker uses the synthetic data to guess information on records in the control dataset.", subsection='Privacy Attacks')
            risk, attack, baseline, control = self.privacy_metrics.execute_singling_out_attack()
            self.report.add_text_to_section('Privacy', "Singling Out Results", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Overall Singling Out {risk}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Main: {attack}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Baseline: {baseline}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Control: {control}", subsection='Privacy Attacks')
            risk, attack, baseline, control = self.privacy_metrics.execute_linkability_attack()
            self.report.add_text_to_section('Privacy', "Linkability Results", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Overall Linkage {risk}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Main: {attack}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Baseline: {baseline}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', f"Control: {control}", subsection='Privacy Attacks')
            self.report.add_text_to_section('Privacy', "Inference Results", subsection='Privacy Attacks')
            self.report.add_figure_to_section('Privacy', self.privacy_metrics.execute_inference_attack(), subtitle='Inference Attack', is_plotly=True, subsection='Privacy Attacks')
        else:
            self.report.add_text_to_section('Privacy', 'Privacy metrics will not be calculated.')

        #### ML Efficacy Section ####
        if self.target_column is not None:
            if ml_efficacy:

                if self.target_type == 'classification':
                    ### Classification Efficacy ###
                    self.report.add_subsection('ML Efficacy', 'Classification Efficacy')
                    self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.assess_efficacy())[0], subtitle='Original Data Classification Report', subsection='Classification Efficacy')
                    self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.assess_efficacy())[1], subtitle='Synthetic Data Classification Report', subsection='Classification Efficacy')
                    self.report.add_figure_to_section('ML Efficacy', (self.ml_efficacy.assess_efficacy())[2], subtitle='ROC Curve', is_plotly=True, subsection='Classification Efficacy')

                elif self.target_type == 'regression':
                    ### Regression Efficacy ###
                    self.report.add_subsection('ML Efficacy', 'Regression Efficacy')
                    self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.assess_efficacy())[0], subtitle='Original Data Regression Report', subsection='Regression Efficacy')
                    self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.assess_efficacy())[1], subtitle='Synthetic Data Regression Report', subsection='Regression Efficacy')
                    self.report.add_figure_to_section('ML Efficacy', (self.ml_efficacy.assess_efficacy())[2], subtitle='Residual Plot', is_plotly=True, subsection='Regression Efficacy')
                
                self.report.add_subsection('ML Efficacy', 'Data Discriminators')
                self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.train_test_discriminator(self.ml_efficacy.original_data, self.ml_efficacy.synthetic_data))[0], subtitle='Data Discriminator Original X Synthetic', subsection='Data Discriminators')
                self.report.add_figure_to_section('ML Efficacy', (self.ml_efficacy.train_test_discriminator(self.ml_efficacy.original_data, self.ml_efficacy.synthetic_data))[2], subtitle='Feature Importance Original X Synthetic', is_plotly=True, subsection='Data Discriminators')
                self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.train_test_discriminator(self.ml_efficacy.original_data, self.ml_efficacy.holdout_data))[0], subtitle='Data Discriminator Original X Holdout', subsection='Data Discriminators')
                self.report.add_figure_to_section('ML Efficacy', (self.ml_efficacy.train_test_discriminator(self.ml_efficacy.original_data, self.ml_efficacy.holdout_data))[2], subtitle='Feature Importance Original X Holdout', is_plotly=True, subsection='Data Discriminators')
                self.report.add_dataframe_to_section('ML Efficacy', (self.ml_efficacy.train_test_discriminator(self.ml_efficacy.synthetic_data, self.ml_efficacy.holdout_data))[0], subtitle='Data Discriminator Synthetic X Holdout', subsection='Data Discriminators')
                self.report.add_figure_to_section('ML Efficacy', (self.ml_efficacy.train_test_discriminator(self.ml_efficacy.synthetic_data, self.ml_efficacy.holdout_data))[2], subtitle='Feature Importance Synthetic X Holdout', is_plotly=True, subsection='Data Discriminators')
            else:
                self.report.add_text_to_section('ML Efficacy', 'ML Efficacy metrics will not be calculated.')
        else:
            print('No target column provided. ML Efficacy metrics will not be calculated.')

        # Generate the html report
        html_report = self.report.generate_report()
        
        print('Report generated.')
        # Write the report to a file
        with open(report_name, 'w') as f:
            f.write(html_report)
        print(f'Report saved to {report_name}')


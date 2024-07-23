import pandas as pd
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class Visualisation:
    def __init__(self, original_data, synthetic_data, holdout_data=None):
        """
        Initialise the Visualisation class with the original and synthetic data.
        :param original_data: Original dataset.
        :param synthetic_data: Synthetic dataset.
        :param holdout_data: Holdout dataset (default is None).
        """
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.holdout_data = holdout_data

    ##############################STATISTICAL SIMILARITY############################################

    def plot_scatter_comparision(self, x, y):
        '''
        Plot a scatter plot comparison between the original and synthetic data.
        :param x: The x-axis column.
        :param y: The y-axis column.
        '''
        dfs = {'Original': self.original_data, 'Synthetic': self.synthetic_data}

        fig = go.Figure()

        for i in dfs:
            fig.add_trace(go.Scatter(x=dfs[i][x], y=dfs[i][y], mode='markers', name=i, opacity=0.7))
        
        fig.update_layout(title=f'{x} vs {y}', xaxis_title=x, yaxis_title=y)
        fig.update_yaxes(type='log')
        fig.update_xaxes(type='log')
        fig.show()

    def plot_descriptive_statistics(self):
        '''
        Plot descriptive statistics for the original and synthetic data. Holdout data is also included if available.
        Creates a line plot with dropdowns for each metric.
        
        return: Plotly figure
        '''
        desc_original = self.original_data.describe()
        desc_synthetic = self.synthetic_data.describe()

        # Transpose the describe dataframes for better plotting
        desc_original_t = desc_original.transpose()
        desc_synthetic_t = desc_synthetic.transpose()

        if self.holdout_data is not None:
            desc_holdout = self.holdout_data.describe().transpose()
            data_sources = {
                'Original': desc_original_t,
                'Synthetic': desc_synthetic_t,
                'Holdout': desc_holdout
            }
        else:
            data_sources = {
                'Original': desc_original_t,
                'Synthetic': desc_synthetic_t
            }

        metrics = desc_original_t.columns.tolist()

        # Create initial figure
        fig = sp.make_subplots(rows=1, cols=1)

        # Add initial traces
        for name, df in data_sources.items():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metrics[0]],
                mode='lines+markers',
                name=f'{name} - {metrics[0]}'
            ))

        # Update layout with dropdowns
        dropdown_buttons = [
            {
                'label': metric,
                'method': 'update',
                'args': [
                    {
                        'y': [df[metric] for df in data_sources.values()],
                        'x': [df.index for df in data_sources.values()],
                        'name': [f'{name} - {metric}' for name in data_sources.keys()]
                    }
                ]
            }
            for metric in metrics
        ]

        fig.update_layout(
            updatemenus=[
                {
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top'
                }
            ],
            title=f'Comparison of Descriptive Statistics',
            xaxis_title='Metrics',
            yaxis_title='Values',
        )
        fig.update_yaxes(type='log')
        return fig

    def plot_descriptive_statistics_radar(self):
        desc_original = self.original_data.describe()
        desc_synthetic = self.synthetic_data.describe()

        # Transpose the describe dataframes for better plotting
        desc_original_t = desc_original.transpose()
        desc_synthetic_t = desc_synthetic.transpose()

        if self.holdout_data is not None:
            desc_holdout = self.holdout_data.describe().transpose()
            data_sources = {
                'Original': desc_original_t,
                'Synthetic': desc_synthetic_t,
                'Holdout': desc_holdout
            }
        else:
            data_sources = {
                'Original': desc_original_t,
                'Synthetic': desc_synthetic_t
            }

        metrics = desc_original_t.columns.tolist()

        # Create initial figure
        fig = sp.make_subplots(rows=1, cols=1)

        # Add initial traces
        for name, df in data_sources.items():
            fig.add_trace(go.Scatterpolar(
                theta=df.index,
                r=df[metrics[0]],
                fill='toself',
                name=f'{name} - {metrics[0]}'
            ))

        # Update layout with dropdowns
        dropdown_buttons = [
            {
                'label': metric,
                'method': 'update',
                'args': [
                    {
                        'r': [df[metric] for df in data_sources.values()],
                        'theta': [df.index for df in data_sources.values()],
                        'name': [f'{name} - {metric}' for name in data_sources.keys()]
                    }
                ]
            }
            for metric in metrics
        ]

        fig.update_layout(
            updatemenus=[
                {
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top'
                }
            ],
            title=f'Comparison of Descriptive Statistics',
        )
        return fig

    def create_plotly_scatter_comparison_with_dropdowns(self, log_scale=False):
        """
        Create a scatter plot comparison with dropdowns for x and y axes.
        :param log_scale: Whether to use a log scale for the axes (default is False). *Bug* log scale not working properly)
        :return: Plotly figure
        """
        if self.holdout_data is not None:
            dfs = {'Original': self.original_data, 'Synthetic': self.synthetic_data, 'Holdout': self.holdout_data}
        else:
            dfs = {'Original': self.original_data, 'Synthetic': self.synthetic_data}

        # Extract column names for options
        x_options = self.original_data.columns.tolist()
        y_options = self.original_data.columns.tolist()

        # Create the initial plot
        fig = go.Figure()

        for i in dfs:
            fig.add_trace(go.Scatter(x=dfs[i][x_options[0]], y=dfs[i][y_options[0]], mode='markers', name=i, opacity=0.7))

        # Create dropdown options for x and y axes
        dropdown_x = [
            {
                'label': x,
                'method': 'update',
                'args': [
                    {'x': [dfs[i][x] for i in dfs]},
                    {
                        'xaxis': {'title': x},
                    }
                ]
            }
            for x in x_options
        ]
        dropdown_y = [
            {
                'label': y,
                'method': 'update',
                'args': [
                    {'y': [dfs[i][y] for i in dfs]},
                    {
                        'yaxis': {'title': y},
                    }
                ]
            }
            for y in y_options
        ]

        # Add dropdowns
        fig.update_layout(
            updatemenus=[
                {
                    'buttons': dropdown_x,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top'
                },
                {
                    'buttons': dropdown_y,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.25,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top'
                }
            ],
            xaxis_title=x_options[0],
            yaxis_title=y_options[0]
        )

        if log_scale:
            fig.update_yaxes(type='log')
            fig.update_xaxes(type='log')

        return fig

    # def plot_distribution_comparison(self, column):
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     sns.histplot(self.original_data[column], ax=ax[0], color='blue', kde=True)
    #     ax[0].set_title('Original Data')
    #     sns.histplot(self.synthetic_data[column], ax=ax[1], color='red', kde=True)
    #     ax[1].set_title('Synthetic Data')
    #     plt.tight_layout()
    #     plt.show()

    # def plot_column_distribution(self, column, nbins=30):
    #     # Calculate KDE for original data
    #     original_kde_x = np.linspace(self.original_data[column].min(), self.original_data[column].max(), 1000)
    #     original_kde_y = stats.gaussian_kde(self.original_data[column])(original_kde_x)
        
    #     # Calculate KDE for synthetic data
    #     synthetic_kde_x = np.linspace(self.synthetic_data[column].min(), self.synthetic_data[column].max(), 1000)
    #     synthetic_kde_y = stats.gaussian_kde(self.synthetic_data[column])(synthetic_kde_x)
        
    #     # Create the histogram for original data
    #     fig = go.Figure()
    #     fig.add_trace(go.Histogram(
    #         x=self.original_data[column],
    #         nbinsx=nbins,
    #         name='Original Data',
    #         marker_color='blue',
    #         opacity=0.5,
    #         histnorm='probability density'
    #     ))

    #     # Create the histogram for synthetic data
    #     fig.add_trace(go.Histogram(
    #         x=self.synthetic_data[column],
    #         nbinsx=nbins,
    #         name='Synthetic Data',
    #         marker_color='red',
    #         opacity=0.5,
    #         histnorm='probability density'
    #     ))

    #     # Add KDE for original data
    #     fig.add_trace(go.Scatter(
    #         x=original_kde_x,
    #         y=original_kde_y,
    #         mode='lines',
    #         name='Original Data KDE',
    #         line=dict(color='blue')
    #     ))

    #     # Add KDE for synthetic data
    #     fig.add_trace(go.Scatter(
    #         x=synthetic_kde_x,
    #         y=synthetic_kde_y,
    #         mode='lines',
    #         name='Synthetic Data KDE',
    #         line=dict(color='red')
    #     ))

    #     # Update layout
    #     fig.update_layout(
    #         title=f'Distribution Comparison for {column}',
    #         xaxis_title=column,
    #         yaxis_title='Density',
    #         barmode='overlay',
    #         legend=dict(x=0.75, y=0.95)
    #     )
    #     fig.show()

    def plot_column_distribution(self, column, nbins=30):
        '''
        Plot the distribution of a single column in the original and synthetic data.
        :param column: The column to plot.
        :param nbins: The number of bins for the histogram (default is 30).
        '''
        # Calculate KDE for original data
        original_kde_x = np.linspace(self.original_data[column].min(), self.original_data[column].max(), 1000)
        original_kde_y = stats.gaussian_kde(self.original_data[column])(original_kde_x)
        
        # Calculate KDE for synthetic data
        synthetic_kde_x = np.linspace(self.synthetic_data[column].min(), self.synthetic_data[column].max(), 1000)
        synthetic_kde_y = stats.gaussian_kde(self.synthetic_data[column])(synthetic_kde_x)

        # Create the histogram for original data
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.original_data[column],
            nbinsx=nbins,
            name='Original Data',
            marker_color='blue',
            opacity=0.5,
            histnorm='probability density'
        ))
        # Create the histogram for synthetic data
        fig.add_trace(go.Histogram(
            x=self.synthetic_data[column],
            nbinsx=nbins,
            name='Synthetic Data',
            marker_color='red',
            opacity=0.5,
            histnorm='probability density'
        ))
        # Add KDE for original data
        fig.add_trace(go.Scatter(
            x=original_kde_x,
            y=original_kde_y,
            mode='lines',
            name='Original Data KDE',
            line=dict(color='blue')
        ))
        # Add KDE for synthetic data
        fig.add_trace(go.Scatter(
            x=synthetic_kde_x,
            y=synthetic_kde_y,
            mode='lines',
            name='Synthetic Data KDE',
            line=dict(color='red')
        ))
        # Update layout
        fig.update_layout(
            title=f'Distribution Comparison for {column}',
            xaxis_title=column,
            yaxis_title='Density',
            barmode='overlay',
            legend=dict(x=0.75, y=0.95)
        )
        fig.show()


    def plot_all_column_distributions(self, nbins=30):
        # Dictionary to store traces for each column
        data_dict = {}

        for column in self.original_data.columns:
            
            # Store traces in dictionary
            data_dict[column] = [
                go.Histogram(
                    x=self.original_data[column],
                    nbinsx=nbins,
                    name='Original Data',
                    marker_color='blue',
                    opacity=0.5,
                    histnorm='probability density'
                ),
                go.Histogram(
                    x=self.synthetic_data[column],
                    nbinsx=nbins,
                    name='Synthetic Data',
                    marker_color='red',
                    opacity=0.5,
                    histnorm='probability density'
                )
            ]

        # Create the figure
        fig = go.Figure()

        # Add the traces for the first column initially
        first_column = self.original_data.columns[0]
        for trace in data_dict[first_column]:
            fig.add_trace(trace)

        # Update layout
        fig.update_layout(
            title=f'Distribution Comparison for {first_column}',
            xaxis_title=first_column,
            yaxis_title='Density',
            barmode='overlay',
            legend=dict(x=0.75, y=0.95)
        )

        # Create dropdown buttons for each column
        dropdown_buttons = []
        for column in self.original_data.columns:
            dropdown_buttons.append(
                {
                    'label': column,
                    'method': 'update',
                    'args': [
                        {'visible': [col == column for col in self.original_data.columns for _ in range(4)]},
                        {'title': f'Distribution Comparison for {column}', 'xaxis': {'title': column}}
                    ]
                }
            )

        # Add dropdown menu to the layout
        fig.update_layout(
            updatemenus=[
                {
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'y': 1.15,
                    'xanchor': 'left',
                    'yanchor': 'top'
                }
            ]
        )

        fig.show()

    def plot_correlation_matrix_array(self, display_output=False):
        """
        Plot the correlation matrices for the original and synthetic data, as well as the difference between them.
        :param display_output: Whether to display the plot (default is False).
        :return: The plot.
        """

        # Calculate the correlation matrices
        corr_original = self.original_data.corr()
        corr_synthetic = self.synthetic_data.corr()
        corr_diff = corr_original - corr_synthetic
        average_diff = np.mean(np.abs(corr_diff.values))

        # Plot the correlation matrices using Plotly
        fig = sp.make_subplots(rows=1, cols=3, subplot_titles=('Original Data Correlation', 'Synthetic Data Correlation', 'Difference in Correlation(Original - Synthetic)'))

        # Create the heatmaps
        # Heatmap 1: Original Data Correlation
        heatmap1 = go.Heatmap(
            z=corr_original.values,
            x=corr_original.columns,
            y=corr_original.index,
            colorscale='Picnic',
            zmin=-1, zmax=1,
            text=np.round(corr_original.values, 2),
            hoverinfo='text'
        )
        fig.add_trace(heatmap1, row=1, col=1)

        # Heatmap 2: Synthetic Data Correlation
        heatmap2 = go.Heatmap(
            z=corr_synthetic.values,
            x=corr_synthetic.columns,
            y=corr_synthetic.index,
            colorscale='Picnic',
            zmin=-1, zmax=1,
            text=np.round(corr_synthetic.values, 2),
            hoverinfo='text'
        )
        fig.add_trace(heatmap2, row=1, col=2)

        # Heatmap 3: Difference in Correlation
        heatmap3 = go.Heatmap(
            z=corr_diff.values,
            x=corr_diff.columns,
            y=corr_diff.index,
            colorscale='Picnic',
            zmin=-1, zmax=1,
            text=np.round(corr_diff.values, 2),
            hoverinfo='text'
        )
        fig.add_trace(heatmap3, row=1, col=3)

        fig.update_layout(height=600, width=1800, title_text="Correlation Matrices")

        if display_output:
            fig.show()
        
        return fig, average_diff.round(2)
    
    # def plot_knn_distance_benchmarks(self, privacy):
    #     # Ensure the distances are pandas Series
    #     benchmark_distances = pd.Series(privacy.benchmark_distances if privacy.benchmark_distances is not None else [])
    #     synthetic_distances = pd.Series(privacy.synthetic_distances if privacy.synthetic_distances is not None else [])
        
    #     # Combine benchmark and synthetic distances into a single DataFrame
    #     distances_data = pd.concat([
    #         pd.DataFrame({'Distance': benchmark_distances, 'Type': 'Benchmark'}),
    #         pd.DataFrame({'Distance': synthetic_distances, 'Type': 'Synthetic'})
    #     ])
        
    #     # Create histogram with KDE using Plotly
    #     fig = px.histogram(distances_data, x='Distance', color='Type', nbins=50,
    #                     barmode='overlay', marginal='box')

    #     # Update the layout
    #     fig.update_layout(title='K-NN Average Distance Distribution',
    #                     xaxis_title='Distance',
    #                     yaxis_title='Frequency')
        
    #     return fig

    ##############################PRIVACY VISUALISATION############################################

    def plot_knn_distance_benchmarks(self, privacy):
        # Ensure the distances are pandas Series
        holdout_X_original_distances = pd.Series(privacy.holdout_x_original_distances if privacy.holdout_x_original_distances is not None else [])
        holdout_X_synthetic_distances = pd.Series(privacy.holdout_x_synthetic_distances if privacy.holdout_x_synthetic_distances is not None else [])

        # Combine benchmark and synthetic distances into a single DataFrame
        distances_data = pd.concat([
            pd.DataFrame({'Distance': holdout_X_original_distances, 'Type': 'Original to Holdout'}),
            pd.DataFrame({'Distance': holdout_X_synthetic_distances, 'Type': 'Synthetic to Holdout'})
        ])

        # Create histogram with KDE using Plotly
        fig = px.histogram(distances_data, x='Distance', color='Type', nbins=50,
                        barmode='overlay', marginal='box')


        # Update the layout
        fig.update_layout(title='k-NN Average Distance Distribution',
                        xaxis_title='Distance',
                        yaxis_title='Frequency')
        
        
        return fig

    def plot_nneighbours_benchmarks(self, privacy):
        # Ensure the distances are pandas Series
        benchmark_distances = pd.DataFrame({'NNeighbours': privacy.nneighbours_benchmark if privacy.nneighbours_benchmark is not None else []})
        # Create histogram with KDE using Plotly
        fig = px.histogram(benchmark_distances, x='NNeighbours', nbins=50,
                        barmode='overlay', marginal='box')

        # Update the layout
        fig.update_layout(title='Average Neighbours Within Threshold Distribution',
                        xaxis_title='NNeighbours',
                        yaxis_title='Frequency')
        
        return fig

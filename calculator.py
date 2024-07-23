import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import plotly.figure_factory as ff

# Function to create the plot
def create_plot(df, x_col, y_col, overlay_point=None):
    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.5, title=f'{x_col} vs {y_col}')
    if overlay_point:
        fig.add_scatter(x=[overlay_point[0]], y=[overlay_point[1]], mode='markers', marker=dict(color='red', size=10), name='Your Company')
    return fig

def create_3d_plot(df, x_col, y_col, z_col, overlay_point=None):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, opacity=0.5, title=f'{x_col} vs {y_col} vs {z_col}')
    if overlay_point:
        fig.add_scatter3d(x=[overlay_point[0]], y=[overlay_point[1]], z=[overlay_point[2]], mode='markers', marker=dict(color='red', size=10), name='Your Company')
    return fig

# Load data
df = pd.read_csv('adult_df.csv')

st.title('Synthetic Data Bivariate Distribution Viewer')

# Column selection
st.sidebar.header('Select Columns')
x_col = st.sidebar.selectbox('X-axis', df.columns)
y_col = st.sidebar.selectbox('Y-axis', df.columns)

# 3D plot option
if st.sidebar.checkbox('3D Plot'):
    z_col = st.sidebar.selectbox('Z-axis', df.columns)

# Subset selection
st.sidebar.header('Subset Selection')
enable_subsetting = st.sidebar.checkbox('Enable Subsetting')

if enable_subsetting:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        categorical_col = st.sidebar.selectbox('Categorical Column', categorical_cols)
        category_value = st.sidebar.selectbox('Category Value', df[categorical_col].unique())

        # Filter dataframe based on selected category
        subset_df = df[df[categorical_col] == category_value]
    else:
        st.sidebar.write("No categorical columns available for subsetting.")
        subset_df = df
else:
    subset_df = df

# Overlay point input
st.sidebar.header('Your Company')
add_overlay = st.sidebar.checkbox('Add Company')

overlay_point = None
user_input_data = {}
selected_features = []

if add_overlay:
    st.sidebar.subheader('Define Company Data')
    for col in subset_df.columns:
        include_col = st.sidebar.checkbox(f'Include {col}', value=True, key=f'include_{col}')
        if include_col:
            selected_features.append(col)
            if subset_df[col].dtype == 'object' or subset_df[col].dtype.name == 'category':
                user_input_data[col] = st.sidebar.selectbox(f'{col} value', subset_df[col].unique(), key=f'overlay_{col}')
            else:
                user_input_data[col] = st.sidebar.number_input(f'{col} value', value=0.0, format="%.2f", key=f'overlay_{col}')
    if x_col in selected_features and y_col in selected_features:
        if 'z_col' in locals() and z_col in selected_features:
            overlay_point = (user_input_data[x_col], user_input_data[y_col], user_input_data[z_col])
        else:
            overlay_point = (user_input_data[x_col], user_input_data[y_col])

# Create and display plot
if 'z_col' in locals():
    fig = create_3d_plot(subset_df, x_col, y_col, z_col, overlay_point)
else:
    fig = create_plot(subset_df, x_col, y_col, overlay_point)
st.plotly_chart(fig)

# Display subset dataframe
if enable_subsetting and 'categorical_col' in locals() and 'category_value' in locals():
    st.write(f'## Subset of Synthetic Data for {categorical_col} = {category_value}')
    st.dataframe(subset_df)
else:
    st.write('## Synthetic Data')
    st.dataframe(subset_df)

# Function to create a mapping for categorical columns
def create_categorical_mapping(df, selected_features):
    mapping = {}
    for col in selected_features:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_values = list(df[col].unique())
            mapping[col] = {value: float(unique_values.index(value)) for value in unique_values}
    return mapping

# Create mapping for categorical columns
categorical_mapping = create_categorical_mapping(subset_df, selected_features)

# Function to transform data using the mapping
def transform_data(df, data, selected_features, mapping):
    transformed_data = {}
    for col in selected_features:
        if col in mapping:
            transformed_data[col] = mapping[col][data[col]]
        else:
            transformed_data[col] = float(data[col])
    return transformed_data

if add_overlay and selected_features:
    transformed_user_input_data = transform_data(subset_df, user_input_data, selected_features, categorical_mapping)
    input_point = np.array(list(transformed_user_input_data.values()))

    def transform_df(df, selected_features, mapping):
        transformed_df = df[selected_features].copy()
        for col in selected_features:
            if col in mapping:
                transformed_df[col] = df[col].apply(lambda x: mapping[col][x])
        return transformed_df
    
    transformed_subset_df = transform_df(subset_df, selected_features, categorical_mapping)
    cov_matrix = np.cov(transformed_subset_df.values.T)
    
    # Regularization to handle singular matrix error
    regularization = 1e-6
    cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
    
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_distr = transformed_subset_df.mean().values

    # Function to calculate Mahalanobis distance for a given point
    def calc_mahalanobis(point, mean_distr, inv_cov_matrix):
        diff = point - mean_distr
        return np.sqrt(diff.dot(inv_cov_matrix).dot(diff.T))

    # Calculate Mahalanobis distance for each point in the subset
    mahalanobis_distances = transformed_subset_df.apply(lambda row: calc_mahalanobis(row.values, mean_distr, inv_cov_matrix), axis=1)
    
    # Calculate Mahalanobis distance for the input point
    input_mahalanobis_dist = calc_mahalanobis(input_point, mean_distr, inv_cov_matrix)
    
    # Calculate the p-value for the Mahalanobis distance
    p_value = 1 - chi2.cdf(input_mahalanobis_dist, transformed_subset_df.shape[1])

    st.write('## Input Company Data')
    st.write(pd.DataFrame([user_input_data]))

    st.write('## Likelihood Calculation')
    st.write(f'Mahalanobis Distance: {input_mahalanobis_dist:.2f}')
    st.write('Mahalanobis distance is a measure of the distance between a point and a distribution.')
    st.write(f'P-value: {p_value:.4f}')
    if p_value < 0.90:
        st.write(':red[The input company is suspicious and warrants further investigation.]')
    else:
        st.write(':green[The input company is likely part of the same distribution as the synthetic data.]')
    st.write(f'This P-value represents the likelihood that the input data point is part of the same distribution as the dataset.')

    # Plot the distribution of Mahalanobis distances with the input point overlay
    fig_dist = px.histogram(mahalanobis_distances, nbins=50, title='Distribution of Mahalanobis Distances For All Points In The Dataset')
    fig_dist.add_vline(x=input_mahalanobis_dist, line_dash="dash", line_color="red", annotation_text="Input Point", annotation_position="top right")
    st.plotly_chart(fig_dist)

    # Determine threshold values
    st.sidebar.header('Determine Outlier Threshold')
    outlier_col = st.sidebar.selectbox('Column to Determine Threshold', selected_features)

    if st.sidebar.button('Calculate Threshold'):

        def find_threshold_for_outlier(df, input_data, col, mean_distr, inv_cov_matrix, mapping, target_p_value=0.10):
            min_value = df[col].min()
            max_value = df[col].max()
            
            negative_threshold_found = False
            positive_threshold_found = False
            step = 100
            negative_threshold = None
            negative_p_value = None
            positive_threshold = None
            positive_p_value = None

            while not negative_threshold_found or not positive_threshold_found:
                if not negative_threshold_found:
                    for value in range(min_value - 1, min_value - step, -1):
                        input_data[col] = value
                        transformed_data = transform_data(df, input_data, selected_features, mapping)
                        input_point = np.array(list(transformed_data.values()))
                        mahalanobis_dist = calc_mahalanobis(input_point, mean_distr, inv_cov_matrix)
                        p_value = 1 - chi2.cdf(mahalanobis_dist, df.shape[1])
                        if p_value <= target_p_value:
                            negative_threshold = value
                            negative_p_value = p_value
                            negative_threshold_found = True
                            break

                if not positive_threshold_found:
                    for value in range(max_value + 1, max_value + step):
                        input_data[col] = value
                        transformed_data = transform_data(df, input_data, selected_features, mapping)
                        input_point = np.array(list(transformed_data.values()))
                        mahalanobis_dist = calc_mahalanobis(input_point, mean_distr, inv_cov_matrix)
                        p_value = 1 - chi2.cdf(mahalanobis_dist, df.shape[1])
                        if p_value <= target_p_value:
                            positive_threshold = value
                            positive_p_value = p_value
                            positive_threshold_found = True
                            break

                step *= 2  # Expand the search range exponentially

            return (negative_threshold, negative_p_value), (positive_threshold, positive_p_value)

        # Example usage
        negative_threshold_info, positive_threshold_info = find_threshold_for_outlier(
            transformed_subset_df, 
            user_input_data.copy(), 
            outlier_col, 
            mean_distr, 
            inv_cov_matrix, 
            categorical_mapping
        )

        if negative_threshold_info[0] is not None:
            st.write(f'Negative threshold for {outlier_col} to be considered an outlier: {negative_threshold_info[0]} (P-value: {negative_threshold_info[1]:.4f})')
        else:
            st.write(f'No negative threshold found for {outlier_col} to be considered an outlier.')

        if positive_threshold_info[0] is not None:
            st.write(f'Positive threshold for {outlier_col} to be considered an outlier: {positive_threshold_info[0]} (P-value: {positive_threshold_info[1]:.4f})')
        else:
            st.write(f'No positive threshold found for {outlier_col} to be considered an outlier.')
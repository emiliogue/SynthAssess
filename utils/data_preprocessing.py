import numpy as np
from sklearn.metrics import pairwise_distances
import gower
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd


def precalculate_distances(data, metric='euclidean'):
    """
    Precalculate distances for the dataset.
    :param data: Dataset for which to calculate distances (Pandas DataFrame).
    :param metric: Distance metric to use (default is 'euclidean').
    :return: Distance matrix (numpy array).
    """
    if metric == 'gower':
        distances = gower.gower_matrix(data)
    else:
        distances = pairwise_distances(data, metric=metric)
    return distances

def encode_dataframes(df1, df2, df3, categorical_cols, ordinal_cols):
    """
    Encodes categorical and ordinal columns in the given dataframes and returns the encoded dataframes along with the encoders.

    Parameters:
    df1 (pd.DataFrame): First dataframe with identical columns.
    df2 (pd.DataFrame): Second dataframe with identical columns.
    df3 (pd.DataFrame): Third dataframe with identical columns.
    categorical_cols (list): List of categorical column names.
    ordinal_cols (list): List of ordinal column names.

    Returns:
    tuple: A tuple containing the encoded dataframes and a dictionary of encoders for each column.
    """
    # Concatenate dataframes to ensure consistent encoding
    combined_df = pd.concat([df1, df2, df3], axis=0)
    
    # Initialize a dictionary to store the encoders
    encoders = {}
    
    # Encode each categorical column
    for col in categorical_cols:
        encoder = LabelEncoder()
        combined_df[col] = encoder.fit_transform(combined_df[col])
        encoders[col] = encoder
    
    # Encode each ordinal column
    for col in ordinal_cols:
        encoder = OrdinalEncoder()
        combined_df[col] = encoder.fit_transform(combined_df[[col]])
        encoders[col] = encoder
    
    # Split the combined dataframe back into the original three dataframes
    encoded_df1 = combined_df.iloc[:len(df1)].reset_index(drop=True)
    encoded_df2 = combined_df.iloc[len(df1):len(df1)+len(df2)].reset_index(drop=True)
    encoded_df3 = combined_df.iloc[len(df1)+len(df2):].reset_index(drop=True)
    
    return encoded_df1, encoded_df2, encoded_df3, encoders

def decode_dataframes(df1, df2, df3, encoders):
    """
    Decodes the encoded dataframes using the provided encoders.

    Parameters:
    df1 (pd.DataFrame): Encoded first dataframe.
    df2 (pd.DataFrame): Encoded second dataframe.
    df3 (pd.DataFrame): Encoded third dataframe.
    encoders (dict): Dictionary of encoders for each column.

    Returns:
    tuple: A tuple containing the decoded dataframes.
    """
    # Combine dataframes for consistent decoding
    combined_df = pd.concat([df1, df2, df3], axis=0)
    
    # Decode each column
    for col, encoder in encoders.items():
        if isinstance(encoder, LabelEncoder):
            combined_df[col] = encoder.inverse_transform(combined_df[col])
        elif isinstance(encoder, OrdinalEncoder):
            combined_df[col] = encoder.inverse_transform(combined_df[[col]])
    
    # Split the combined dataframe back into the original three dataframes
    decoded_df1 = combined_df.iloc[:len(df1)].reset_index(drop=True)
    decoded_df2 = combined_df.iloc[len(df1):len(df1)+len(df2)].reset_index(drop=True)
    decoded_df3 = combined_df.iloc[len(df1)+len(df2):].reset_index(drop=True)
    
    return decoded_df1, decoded_df2, decoded_df3
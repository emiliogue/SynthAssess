def verify_input_parameters(original_data, synthetic_data, holdout_data=None, ignore_data_type=False):
    """
    Verify that the input parameters are correct.
    """
    # Check if the data is a pandas dataframe
    if not hasattr(original_data, 'columns'):
        raise ValueError('Original data must be a pandas dataframe.')
    if not hasattr(synthetic_data, 'columns'):
        raise ValueError('Synthetic data must be a pandas dataframe.')
    if holdout_data is not None and not hasattr(holdout_data, 'columns'):
        raise ValueError('Holdout data must be a pandas dataframe.')
    # Check if the dataframes have the same columns
    if not original_data.columns.equals(synthetic_data.columns):
        raise ValueError('Original and synthetic data must have the same columns.')
    if holdout_data is not None and not original_data.columns.equals(holdout_data.columns):
        raise ValueError('Original and holdout data must have the same columns.')
    # Check if the dataframes have the same data types
    # if not original_data.dtypes.equals(synthetic_data.dtypes):
    #     raise ValueError('Original and synthetic data must have the same data types.')
    # if holdout_data is not None and not original_data.dtypes.equals(holdout_data.dtypes):
    #     raise ValueError('Original and holdout data must have the same data types.')
    pass
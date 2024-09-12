import pandas as pd

def load_exposure(file_path):
    """
    Load the exposure data from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing the exposure data.

    Returns:
    pd.DataFrame: DataFrame containing the exposure data.
    """
    exposure_df = pd.read_csv(file_path)
    return exposure_df
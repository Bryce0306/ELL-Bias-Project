import os
import pandas as pd


def load_data_from_folder(folder_path: str, file_extension: str = '.csv') -> pd.DataFrame:
    """
    Load and concatenate all data files from a specified folder.

    Args:
        folder_path (str): Path to the folder containing the data files.
        file_extension (str): File extension of the data files to load (default: '.csv').

    Returns:
        pd.DataFrame: A single DataFrame containing the combined data from all matching files.

    Raises:
        ValueError: If the folder does not exist or no matching files are found.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")

    # Find all files in the folder with the specified extension
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(file_extension)]
    
    # Raise an error if no matching files are found
    if not files:
        raise ValueError(f"No files with extension '{file_extension}' found in folder '{folder_path}'.")

    # Load each file into a DataFrame and concatenate them
    dataframes = [pd.read_csv(file) for file in files]
    return pd.concat(dataframes, ignore_index=True)

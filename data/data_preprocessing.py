import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


def classify_ell(value):
    """
    Classify the English Language Acquisition (ELA) value into categories:
        - 'EL': English Learners, includes 'EL' (English Learner) and 'ADEL' (Adult English Learner).
        - 'Non-EL': Non-English Learners, includes 'EO' (English Only), 'RFEP' (Reclassified Fluent English Proficient),
           and 'IFEP' (Initial Fluent English Proficient).
        - 'Exclude': Includes 'TBD' (To Be Determined) and 'EPU' (English Proficiency Unknown).
        - 'Unknown': For values not in the above categories.

    Args:
        value (str): The ELA value to classify.
    
    Returns:
        str: The corresponding category ('EL', 'Non-EL', 'Exclude', or 'Unknown').
    """
    if value in ['EL', 'ADEL']:
        return 'EL'
    elif value in ['EO', 'RFEP', 'IFEP']:
        return 'Non-EL'
    elif value in ['TBD', 'EPU']:
        return 'Exclude'
    else:
        return 'Unknown'
    

def balance_group_sizes(df: pd.DataFrame, group_column: str, min_size: int) -> pd.DataFrame:
    """
    Ensure each group in the DataFrame has at least `min_size` rows.
    If a group has fewer rows, additional samples are added to meet the minimum size.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_column (str): The column name used for grouping.
        min_size (int): The minimum number of rows required for each group.

    Returns:
        pd.DataFrame: A new DataFrame with balanced group sizes.
    """
    # Group the DataFrame by the specified column
    grouped = df.groupby(group_column)

    # Process each group and ensure the size requirement is met
    balanced_groups = []
    for group_name, group_data in grouped:
        if len(group_data) < min_size:
            # Calculate how many additional rows are needed
            additional_samples = group_data.sample(
                n=min_size - len(group_data),
                replace=True,    # Allow duplicate rows
                random_state=42  # Ensure reproducibility
            )
            group_data = pd.concat([group_data, additional_samples], ignore_index=True)
        
        # Add the balanced group to the list
        balanced_groups.append(group_data)

    # Combine all balanced groups into a single DataFrame
    balanced_df = pd.concat(balanced_groups, ignore_index=True)
    return balanced_df


def balance_and_sample(df: pd.DataFrame, group_column: str, min_samples_per_group: int, total_size: int) -> pd.DataFrame:
    """
    Balance groups in the dataset by ensuring a minimum number of samples per group
    and adjust the total dataset size to the specified target size.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_column (str): The column used for grouping.
        min_samples_per_group (int): Minimum number of samples required for each group.
        total_size (int): The total size of the resulting dataset.

    Returns:
        pd.DataFrame: A balanced and sampled DataFrame.
    """
    # Step 1: Ensure minimum samples per group
    grouped = df.groupby(group_column)
    balanced_groups = grouped.apply(
        lambda group: group.sample(n=min_samples_per_group, replace=True, random_state=42)
    ).reset_index(drop=True)

    # Step 2: Calculate the number of samples still needed
    remaining_samples = total_size - len(balanced_groups)

    if remaining_samples > 0:
        # Step 3: Sample additional rows if necessary
        remaining_data = df.loc[~df.index.isin(balanced_groups.index)]
        additional_samples = remaining_data.sample(
            n=remaining_samples,
            replace=True if len(remaining_data) < remaining_samples else False,
            random_state=42
        )
        result = pd.concat([balanced_groups, additional_samples], ignore_index=True)
    else:
        # Step 4: Trim the data if it exceeds the required size
        result = balanced_groups.sample(n=total_size, random_state=42).reset_index(drop=True)

    return result


def preprocess_data(df):
    """
    Preprocess the raw DataFrame by renaming columns, extracting text, 
    and classifying English Language Learners (ELL).

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    data = df[['ScoreSE', 'EnglishLanguageAcquisition', 'REPLACE(SUBSTR(CR_TEXT1||CR_TEXT2,1,37000),CHR(10))']] \
        .rename(columns={
            'ScoreSE': 'score',
            'REPLACE(SUBSTR(CR_TEXT1||CR_TEXT2,1,37000),CHR(10))': 'text'
        })
    data['text'] = data['text'].str.extract(r'<value>(.*?)</value>')
    data['ELL Classification'] = data['EnglishLanguageAcquisition'].apply(classify_ell)
    data = data.drop(columns=['EnglishLanguageAcquisition'])
    return data


def preprocess_dataframe(df):
    """
    Preprocess a DataFrame by processing labels and cleaning text data.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'LABEL_COLUMN' and 'DATA_COLUMN'.
    """
    df['score'] = pd.Categorical(df['score'], ordered=True)
    df['score_codes'] = df['score'].cat.codes

    df['text'] = df['text'].replace(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', regex=True
    )

    df = df.rename(columns={'score': 'LABEL_COLUMN', 'text': 'DATA_COLUMN'})
    return df[['LABEL_COLUMN', 'DATA_COLUMN']]


def build_groups(data, min_samples_per_class, test_size=0.2):
    """
    Build groups for training and testing directly from the input data.
    This function includes preprocessing, splitting, balancing, and sampling logic.

    Args:
        data (pd.DataFrame): The input raw data.
        min_samples_per_class (int): Minimum number of samples required per class.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        list: A list of (train, test) pairs, ready for model training.
    """
    groups = []

    # Split the data by ELL classification
    ell_data = data[data['ELL Classification'] == 'EL'].drop(columns=['ELL Classification'])
    non_ell_data = data[data['ELL Classification'] == 'Non-EL'].drop(columns=['ELL Classification'])

    # Split each group into training and testing sets
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=42)
    ell_train, ell_test = train_test_split(ell_data, test_size=test_size, random_state=42)
    non_ell_train, non_ell_test = train_test_split(non_ell_data, test_size=test_size, random_state=42)

    # Balance the training sets
    ell_train_balanced = balance_group_sizes(ell_train, group_column='score', min_size=min_samples_per_class)
    non_ell_train_balanced = balance_group_sizes(non_ell_train, group_column='score', min_size=min_samples_per_class)

    min_size = min(len(ell_train_balanced), len(non_ell_train_balanced))
    ell_train_sampled = balance_and_sample(ell_train_balanced, 'score', min_samples_per_class, min_size)
    non_ell_train_sampled = balance_and_sample(non_ell_train_balanced, 'score', min_samples_per_class, min_size)

    # Combine the balanced training and testing sets
    complete_train = pd.concat([ell_train_sampled, non_ell_train_sampled], ignore_index=True)
    complete_test = pd.concat([ell_test, non_ell_test], ignore_index=True)

    groups.append((preprocess_dataframe(complete_train), preprocess_dataframe(complete_test)))
    groups.append((preprocess_dataframe(ell_train_sampled), preprocess_dataframe(ell_test)))
    groups.append((preprocess_dataframe(non_ell_train_sampled), preprocess_dataframe(non_ell_test)))

    return groups


class CustomTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = list(dataframe['DATA_COLUMN'])
        self.labels = list(dataframe['LABEL_COLUMN'])
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding['token_type_ids'].squeeze()
        return {
            'input_ids': input_ids.to(torch.long),
            'attention_mask': attention_mask.to(torch.long),
            'token_type_ids': token_type_ids.to(torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
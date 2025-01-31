# Model Bias on English Language Learners

This project investigates potential bias introduced by imbalanced training data when fine-tuning BERT-based models in an educational assessment context. Specifically, we compare English Language Learners (ELL) and non-English Language Learners (Non-ELL) in terms of model performance and propose the **Mean Score Gap (MSG)** metric to quantify any observed performance gap.

---

## Table of Contents

- [Model Bias on English Language Learners](#model-bias-on-english-language-learners)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Project Structure](#project-structure)
  - [Environment and Dependencies](#environment-and-dependencies)
  - [Data Preparation](#data-preparation)
  - [Usage](#usage)
  - [Key Components](#key-components)
    - [1. `main.py`](#1-mainpy)
    - [2. `models/BERT.py`](#2-modelsbertpy)
    - [3. `data/data_preprocessing.py`](#3-datadata_preprocessingpy)
    - [4. `data/data_loader.py`](#4-datadata_loaderpy)
    - [5. `utils/metrics.py`](#5-utilsmetricspy)
    - [6. `utils/tools.py`](#6-utilstoolspy)
  - [Experimental Workflow](#experimental-workflow)
  - [Evaluation Metrics](#evaluation-metrics)
  - [References and Acknowledgments](#references-and-acknowledgments)

---

## Background

- **Motivation**  
  In educational assessment, large language models (e.g., BERT) are commonly fine-tuned to automatically score student responses. However, these training datasets often exhibit an imbalance—for instance, fewer ELL students than Non-ELL students—which may cause unintended bias in the resulting model.
  
- **Research Question**  
  - Does the imbalance in ELL vs. Non-ELL data propagate as bias into the model’s predictions?  
  - We conduct comparative experiments by training on ELL, Non-ELL, and Mixed datasets and introduce **Mean Score Gap (MSG)** to measure how differently the model assesses these groups.

---

## Project Structure

```plaintext
ell_bias_new/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py        # Loads and merges CSV files into a single DataFrame
│   └── data_preprocessing.py # Preprocessing logic, custom Dataset, etc.
│
├── datasets/
│   └── VH811932/
│       ├── merged_CAST_2018_2019_VH811932.csv
│       └── merged_CAST_2020_2021_VH811932.csv
│       # Put your CSV data here
│
├── models/
│   ├── __init__.py
│   └── BERT.py               # Training and inference methods for BERT
│
├── predictions/
│   # Prediction logs (CSV) are saved here after inference
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # Functions for compute_accuracy, compute_msg, etc.
│   └── tools.py              # evaluate_results and other utility functions
│
├── main.py                   # Entry point for the overall pipeline
├── training.log              # Log file (auto-generated during training)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Environment and Dependencies

- **Python Version**: 3.10.12  
- **Key Dependencies**:
  - `torch == 2.5.0`  
  - `transformers == 4.48.1`  
  - `pandas == 2.2.3`  
  - `numpy == 2.2.2`  
  - `scikit-learn == 1.6.1`  
  - `scipy == 1.15.1`  
  - `tqdm == 4.67.1`  
  - `matplotlib == 3.10.0`  
  - `safetensors == 0.5.2`

Install the dependencies by running:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

1. **Data Placement**  
   - Place your dataset CSV files in the `datasets/VH811932/` directory.
   - For example:
     - `merged_CAST_2018_2019_VH811932.csv`
     - `merged_CAST_2020_2021_VH811932.csv`
   - If you are using a different directory or file naming scheme, update the `CONFIG["data_folder"]` and `CONFIG["file_extension"]` in `main.py` accordingly.

2. **Expected Columns**  
   - The following columns are required for the dataset to be processed:
     - `ScoreSE`: This column contains the target scores (used as labels for training and evaluation).
     - `EnglishLanguageAcquisition`: Indicates the student's language acquisition status, such as:
       - `EL`: English Learners
       - `ADEL`: Adult English Learners
       - `EO`: English Only
       - `RFEP`: Reclassified Fluent English Proficient
       - `IFEP`: Initial Fluent English Proficient
       - `TBD`: To Be Determined
       - `EPU`: English Proficiency Unknown
     - `CR_TEXT1 || CR_TEXT2`: Combined text responses for each student. The code extracts text from `<value>` tags in this column.

3. **Preprocessing Steps**  
   - The preprocessing logic in `data_preprocessing.py` performs the following steps:
     - **Classification**: Groups students into `ELL`, `Non-ELL`, `Exclude`, or `Unknown` categories based on the `EnglishLanguageAcquisition` column.
     - **Balancing**: Ensures each group has a minimum number of samples (`min_samples_per_group`) by oversampling as needed.
     - **Splitting**: Divides the dataset into `train` and `test` subsets based on the `test_size` parameter (default is 20%).
     - **Feature Engineering**: Extracts and cleans textual data, standardizes column names, and encodes categorical scores into numeric labels.

4. **Output**  
   - The preprocessed data is split into three groups:
     - **Mixed**: Contains both ELL and Non-ELL data.
     - **ELL-only**: Contains data exclusively for ELL students.
     - **Non-ELL-only**: Contains data exclusively for Non-ELL students.
   - Each group is further split into training and testing datasets, ensuring a balanced representation of target labels.

---

## Usage

1. **Adjust Configurations (Optional)**  
   - Before running the script, you can modify the configurations in the `CONFIG` dictionary located in `main.py`.  
   - Example:
     ```python
     CONFIG: Dict[str, Any] = {
         "data_folder": "datasets/VH811932",     # Path to the dataset folder
         "file_extension": ".csv",              # File extension of data files
         "test_size": 0.2,                      # Proportion of test data
         "min_samples_per_class": 4,            # Minimum samples per class for balancing
         "model_name": "bert-base-uncased",     # Pre-trained BERT model
         "batch_size": 8,                       # Training batch size
         "learning_rate": 3e-5,                 # Learning rate for optimizer
         "weight_decay": 0.02,                  # Weight decay for optimizer
         "warmup_ratio": 0.4,                   # Ratio for learning rate warmup
         "num_epochs": 50,                      # Total number of training epochs
         "early_stop_patience": 8,              # Early stopping patience
         "early_stop_start_epoch": 5,           # When to start checking for early stopping
         "max_seq_length": 100,                 # Maximum sequence length for tokenization
         "prediction_batch_size": 32,           # Batch size for predictions
         "output_folder": "predictions",        # Directory to save prediction outputs
         "device_id": 0                         # GPU device ID (set to 0 for the first GPU)
     }
     ```

2. **Run the Main Script**  
   - To execute the full pipeline (data loading, preprocessing, training, inference, and evaluation), simply run:
     ```bash
     python main.py
     ```
   - The script will:
     - Load and preprocess the data.
     - Train three separate BERT models on ELL, Non-ELL, and Mixed datasets.
     - Generate predictions for the corresponding test datasets.
     - Save the predictions as CSV files and evaluate the models.

3. **Outputs**  
   - **Logs**:
     - Training logs are recorded in `training.log` and include details about loss, accuracy, and early stopping.
   - **Predictions**:
     - Prediction results are saved as CSV files in the `predictions/` directory. Each file is named based on the model and test set combination, e.g., `model_mixed_predict_ell_test.csv`.
   - **Evaluation**:
     - The console output includes:
       - Accuracy for each model-test combination.
       - Mean Score Gap (MSG) between ELL and Non-ELL predictions.
       - Optional statistical tests (e.g., T-tests) to compare model performance.

4. **Reproducing Results**  
   - Ensure the `data/` folder contains the same dataset used in the original experiment.
   - Use the same random seed (`42` by default) to reproduce training and evaluation results.

5. **Key Notes**  
   - Ensure your system has GPU support if training on large datasets.
   - Adjust the `batch_size` and `num_epochs` based on your system’s memory capacity and project requirements.
   - For detailed debug logs, modify the logging level in `main.py`:
     ```python
     logging.basicConfig(
         level=logging.DEBUG,
         format="%(asctime)s [%(levelname)s] %(message)s",
         datefmt="%Y-%m-%d %H:%M:%S"
     )
     ```

By following these steps, you can run the entire pipeline and evaluate the model's performance effectively.

---

## Key Components

### 1. `main.py`
- **Purpose**: Acts as the entry point for the entire pipeline.
- **Responsibilities**:
  - Loads data from the specified folder.
  - Preprocesses the data (e.g., balancing and sampling).
  - Trains three BERT models on ELL, Non-ELL, and Mixed datasets.
  - Generates predictions for the test datasets.
  - Evaluates the models using accuracy, MSG, and optional statistical tests.

---

### 2. `models/BERT.py`
- **`train_BERT_model`**:
  - Fine-tunes a pre-trained BERT model for sequence classification.
  - Implements early stopping, warmup, and weight decay.
  - Handles training and validation processes.
- **`predict_BERT_models`**:
  - Uses trained models to predict on test datasets.
  - Optionally saves the predictions as CSV files.

---

### 3. `data/data_preprocessing.py`
- Handles all preprocessing steps:
  - **`classify_ell`**: Classifies students into ELL or Non-ELL categories.
  - **Balancing and Sampling**: Ensures fair representation of each class in training data.
  - **`CustomTextDataset`**: Converts preprocessed data into a PyTorch-compatible dataset.
  - **`build_groups`**: Splits the data into ELL, Non-ELL, and Mixed subsets for training and testing.

---

### 4. `data/data_loader.py`
- **`load_data_from_folder`**:
  - Loads and merges CSV files from the specified directory into a single Pandas DataFrame.

---

### 5. `utils/metrics.py`
- **`compute_accuracy`**:
  - Calculates accuracy based on label-prediction pairs.
- **`compute_msg`**:
  - Computes Mean Score Gap (MSG) between:
    - Two datasets (e.g., ELL vs. Non-ELL) in `compare` mode.
    - Model predictions and human labels in `human` mode.

---

### 6. `utils/tools.py`
- **`evaluate_results`**:
  - Aggregates predictions across all models and test sets.
  - Computes accuracy and MSG metrics.
  - Optionally performs T-tests for statistical significance.

---

## Experimental Workflow

1. **Data Loading and Preprocessing**
   - Read multiple CSV files from the dataset folder.
   - Extract text, clean columns, and classify ELL/Non-ELL groups.
   - Balance and split the data into training and testing sets.

2. **Model Training**
   - Train separate BERT models on:
     - Mixed dataset
     - ELL-only dataset
     - Non-ELL-only dataset
   - Validate models on corresponding test sets.
   - Implement early stopping to prevent overfitting.

3. **Inference**
   - Use trained models to generate predictions for:
     - Mixed test set
     - ELL-only test set
     - Non-ELL-only test set
   - Save predictions as CSV files in the `predictions/` folder.

4. **Evaluation**
   - **Accuracy**:
     - Compute accuracy for each model-test combination.
   - **MSG (Mean Score Gap)**:
     - Quantify differences in model predictions for ELL vs. Non-ELL groups.
   - **Statistical Significance**:
     - Perform T-tests to assess whether performance differences are statistically significant.

5. **Results Interpretation**
   - Analyze accuracy and MSG metrics to understand potential biases in the models.
   - Use visualizations or logs to present findings.

This structured workflow ensures reproducibility and a comprehensive evaluation of model bias on ELL and Non-ELL datasets.

---

## Evaluation Metrics

1. **Accuracy**  
   - Measures the proportion of correct predictions:
     \[
     \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
     \]
   - Computed for each model-test set combination (Mixed, ELL, Non-ELL).

2. **Mean Score Gap (MSG)**  
   - Quantifies the difference between predictions for different groups or between predictions and human labels:
     - **Compare Mode**: Measures the absolute difference in the mean predictions between two groups (e.g., ELL vs. Non-ELL).
       \[
       \text{MSG}_{\text{compare}} = |\text{Mean(Group A)} - \text{Mean(Group B)}|
       \]
     - **Human Mode**: Measures the gap between predictions and human-assigned labels in the same group.
       \[
       \text{MSG}_{\text{human}} = |\text{Mean(Predictions)} - \text{Mean(Human Labels)}|
       \]

3. **Statistical Significance (Optional)**  
   - Uses **T-tests** to compare distributions of accuracies or MSG values across models or groups.
   - Reports:
     - **T-statistic**: Indicates the magnitude of difference between groups.
     - **P-value**: Determines if the difference is statistically significant (e.g., \( p < 0.05 \)).

4. **Outputs**  
   - Metrics are printed to the console during evaluation and include:
     - Accuracy for each model-test pair.
     - MSG for ELL vs. Non-ELL and human vs. model predictions.
     - T-test results for performance differences.

---

## References and Acknowledgments

- **Tools and Frameworks**:
  - [Hugging Face Transformers](https://github.com/huggingface/transformers): For BERT fine-tuning and tokenization.
  - [PyTorch](https://pytorch.org/): For building and training deep learning models.
  - [Scikit-learn](https://scikit-learn.org/): For data preprocessing and evaluation metrics.
  - [SciPy](https://scipy.org/): For statistical testing and numerical computations.

- **Dataset**:
  - The project relies on student response data, including scores and textual responses, for assessing model bias in educational contexts.

- **Contributors**:
  - UGA School of Computing

If you find this project useful or have any questions, feel free to contact the authors. Thank you for supporting this initiative!
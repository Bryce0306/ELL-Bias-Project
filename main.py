import os
import warnings
import logging
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
import random

from data.data_loader import load_data_from_folder
from data.data_preprocessing import preprocess_data, build_groups
from models.BERT import train_BERT_model, predict_BERT_models
from utils.tools import evaluate_results

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ========== CONFIGURATION ===========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

CONFIG: Dict[str, Any] = {
    "data_folder": "datasets/VH811932",
    "file_extension": ".csv",
    "test_size": 0.2,
    "min_samples_per_class": 4,
    "model_name": "bert-base-uncased",
    "batch_size": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.02,
    "warmup_ratio": 0.4,
    "num_epochs": 50,
    "early_stop_patience": 8,
    "early_stop_start_epoch": 5,
    "max_seq_length": 100,
    "prediction_batch_size": 32,
    "output_folder": "predictions",
    "device_id": 0
}


# ========== MAIN WORKFLOW ===========
def main() -> None:
    """
    Main function:
    1) Load data
    2) Preprocess data
    3) Build data groups
    4) Train multiple BERT models
    5) Predict with the trained models and save outputs
    6) Evaluate predict results
    """
    set_seed()

    # Determine device: GPU if available, else CPU
    device_str = f"cuda:{CONFIG['device_id']}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # 1) Load data
    data_folder = CONFIG["data_folder"]
    file_extension = CONFIG["file_extension"]

    if not os.path.exists(data_folder):
        logger.error(f"Data folder '{data_folder}' does not exist.")
        raise FileNotFoundError(f"Data folder '{data_folder}' does not exist.")
    
    try:
        data = load_data_from_folder(data_folder, file_extension=file_extension)
        logger.info(f"Data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data from folder: {data_folder}. Error: {e}")
        raise  # Re-raise so the program halts if data loading fails

    # 2) Preprocess data
    try:
        data = preprocess_data(data)
        unique_labels = data["score"].unique().tolist()
        logger.info(f"Unique labels: {unique_labels}")
    except Exception as e:
        logger.error(f"Failed to preprocess data. Error: {e}")
        raise

    # 3) Build data groups
    try:
        groups = build_groups(
            data,
            min_samples_per_class=CONFIG["min_samples_per_class"],
            test_size=CONFIG["test_size"]
        )
        logger.info(f"{len(groups)} data group(s) have been created for training and testing.")
    except Exception as e:
        logger.error(f"Failed to build data groups. Error: {e}")
        raise

    # 4) Train models
    trained_models = []
    for idx, (train_dataset, test_dataset) in enumerate(groups):
        try:
            logger.info(f"Training model {idx + 1}/{len(groups)}...")
            model_info = train_BERT_model(
                train_data=train_dataset,
                test_data=test_dataset,
                model_name=CONFIG["model_name"],
                num_labels=len(unique_labels),
                batch_size=CONFIG["batch_size"],
                learning_rate=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
                warmup_ratio=CONFIG["warmup_ratio"],
                num_epochs=CONFIG["num_epochs"],
                early_stop_patience=CONFIG["early_stop_patience"],
                early_stop_start_epoch=CONFIG["early_stop_start_epoch"],
                max_seq_length=CONFIG["max_seq_length"],
                device=device
            )
            trained_models.append(model_info)
            logger.info(f"Model {idx + 1} training completed.")
        except Exception as e:
            logger.error(f"Failed to train model {idx + 1}. Error: {e}")
            raise

    # 5) Predict and save results
    output_folder = CONFIG["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    best_models = [m["best_model"] for m in trained_models]
    logger.info("Starting inference with all trained models...")
    try:
        results = predict_BERT_models(
            models=best_models,
            groups=groups,
            model_name=CONFIG["model_name"],
            max_length=CONFIG["max_seq_length"],
            batch_size=CONFIG["prediction_batch_size"],
            device=device,
            output_dir=output_folder,
            save_to_file=True
        )
        logger.info(f"Inference complete. Predictions saved to: {output_folder}")
    except Exception as e:
        logger.error(f"Failed during prediction. Error: {e}")
        raise

    # 6) Evaluate predict results
    evaluate_results(results)


# ========== ENTRY POINT ===========
if __name__ == "__main__":
    main()

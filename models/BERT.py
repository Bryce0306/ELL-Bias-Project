import torch
import copy
import pandas as pd
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import logging
from typing import List, Dict
from data.data_preprocessing import CustomTextDataset


def train_BERT_model(train_data, test_data, 
                model_name='bert-base-uncased', 
                num_labels=None, 
                batch_size=8, 
                learning_rate=3e-5, 
                weight_decay=0.02, 
                warmup_ratio=0.4, 
                num_epochs=50, 
                early_stop_patience=8, 
                early_stop_start_epoch=5, 
                max_seq_length=100, 
                device=torch.device("cuda:0")):
    """
    Train a BERT model for sequence classification with early stopping and parameterized settings.

    Args:
        train_data (CustomTextDataset): Training dataset.
        test_data (CustomTextDataset): Testing dataset.
        model_name (str): Pretrained model name from HuggingFace's transformers library.
        num_labels (int): Number of output labels for classification.
        batch_size (int): Batch size for DataLoader.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for AdamW optimizer.
        warmup_ratio (float): Ratio of warmup steps for the learning rate scheduler.
        num_epochs (int): Total number of epochs for training.
        early_stop_patience (int): Number of epochs with no improvement before early stopping.
        early_stop_start_epoch (int): Epoch from which to start early stopping checks.
        max_seq_length (int): Maximum sequence length for tokenization.
        device (torch.device): Device to run the training on (e.g., "cuda" or "cpu").

    Returns:
        dict: A dictionary containing the best model, training losses, validation accuracies, and the best accuracy.
    """
    # Initialize tokenizer and DataLoaders
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    train_dataset = CustomTextDataset(train_data, tokenizer, max_seq_length)
    test_dataset = CustomTextDataset(test_data, tokenizer, max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # Set optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, 
        num_warmup_steps=int(warmup_ratio * total_steps), 
        num_training_steps=total_steps
    )

    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Logging setup
    logging.basicConfig(filename='training.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')

    # Training variables
    best_model = None
    best_accuracy = 0
    epochs_no_improve = 0
    train_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}: Training started.")
        print(f"\n======== Epoch {epoch + 1}/{num_epochs} ========")
        print("Training...")

        model.train()
        total_loss = 0

        # Training step
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch + 1}: Average Training Loss = {avg_train_loss}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        total_eval_accuracy = 0

        for batch in validation_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    token_type_ids=batch['token_type_ids'].to(device),
                    labels=batch['labels'].to(device)
                )
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            accuracy = (preds == batch['labels'].to(device)).cpu().numpy().mean() * 100
            total_eval_accuracy += accuracy

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        val_accuracies.append(avg_val_accuracy)
        logging.info(f"Epoch {epoch + 1}: Validation Accuracy = {avg_val_accuracy}")
        print(f"Validation Accuracy: {avg_val_accuracy:.2f}%")

        # Check for improvement and early stopping
        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
            logging.info(f"Epoch {epoch + 1}: Validation accuracy improved. Best model updated.")
        elif epoch >= early_stop_start_epoch:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    return {
        'best_model': best_model,
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }


def predict_BERT_models(models: List,
                        groups: List,
                        model_name='bert-base-uncased',
                        max_length=100,
                        batch_size=32,
                        device=torch.device("cuda:0"),
                        output_dir="predictions/",
                        save_to_file=False) -> Dict[str, List[tuple]]:
    """
    Run predictions for multiple models and keep results in memory, with optional CSV saving.

    Args:
        models (list): List of trained models.
        groups (list): List of (train, test) pairs corresponding to each model.
        model_name (str): Name of the pretrained BERT model for tokenization.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for the DataLoader.
        device (torch.device): The device for running predictions (e.g., "cuda" or "cpu").
        output_dir (str): Directory to save prediction results (if save_to_file=True).
        save_to_file (bool): Whether to save prediction results to CSV files.

    Returns:
        None
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    model_labels = ["mixed", "ell", "non_ell"]
    test_data_labels = ["mixed", "ell", "non_ell"]

    results = {}

    for model_idx, (test_model, model_label) in enumerate(zip(models, model_labels)):
        test_sets = [groups[0][1], groups[1][1], groups[2][1]]

        for test_data, test_label in zip(test_sets, test_data_labels):
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
            test_dataset = CustomTextDataset(test_data, tokenizer, max_length)

            validation_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            predict_log =[]

            for batch in validation_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                with torch.no_grad():
                    outputs = test_model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         labels=labels)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()

                for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    predict_log.append((label, pred))
                    
            key = f"model_{model_label}_test_{test_label}"
            results[key] = predict_log

            if save_to_file:
                df = pd.DataFrame(predict_log, columns=['labels', 'preds'])
                file_path = os.path.join(
                    output_dir,
                    f"model_{model_label}_predict_{test_label}_data_log.csv"
                )
                df.to_csv(file_path, index=False)

                print(f"Model trained on '{model_label}' data has generated predictions "
                      f"for the '{test_label}' test set. Results saved to: {file_path}")
            else:
                print(f"Model '{model_label}' predictions for test set '{test_label}' completed.")

    return results

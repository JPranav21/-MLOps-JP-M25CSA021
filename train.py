import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import create_dataloader, format_time
import random
import time

def check_gpu():
    """Check for GPU availability for TensorFlow and PyTorch."""
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        print('TensorFlow found GPU at:', device_name)
    else:
        print('TensorFlow: GPU not found')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'PyTorch using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
        print('PyTorch using CPU')
    return device

def load_dataset(path="./cola_public/raw/in_domain_train.tsv"):
    """Load CoLA dataset and return sentences and labels."""
    df = pd.read_csv(path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    print(f"Number of training sentences: {df.shape[0]:,}")
    return df.sentence.values, df.label.values

def tokenize_and_pad(sentences, max_len=64):
    """Tokenize sentences using BERT tokenizer and pad sequences."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[int(tok_id > 0) for tok_id in seq] for seq in input_ids]
    return input_ids, attention_masks, tokenizer

def split_data(input_ids, attention_masks, labels, test_size=0.1, random_state=42):
    """Split data into training and validation sets."""
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=random_state, test_size=test_size)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, random_state=random_state, test_size=test_size)
    return train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks

def build_model(num_labels=2):
    """Load BERT model for sequence classification."""
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    return model

def prepare_optimizer_scheduler(model, train_dataloader, epochs=4, lr=2e-5, eps=1e-8):
    """Create AdamW optimizer and linear learning rate scheduler."""
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return optimizer, scheduler

def set_seed(seed_val=42):
    """Set random seed for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

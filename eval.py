import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from utils import create_dataloader

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

def plot_training_loss(loss_values):
    """Plot training loss curve."""
    plt.plot(loss_values, 'b-o')
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def prepare_test_dataloader(sentences, labels, tokenizer, max_len=64, batch_size=32):
    """Tokenize, pad, mask, and create dataloader for test set."""
    from keras.preprocessing.sequence import pad_sequences

    input_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(tok_id>0) for tok_id in seq] for seq in input_ids]

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    return prediction_dataloader

def evaluate_model(model, dataloader, device):
    """Evaluate model and return Matthews correlation coefficient."""
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    return mcc

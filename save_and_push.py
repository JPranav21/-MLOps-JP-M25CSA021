import os
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from huggingface_hub import login

def save_model_locally(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str = './model_save/'):
    """Save fine-tuned model and tokenizer locally."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved locally to {output_dir}")
    return output_dir

def push_model_to_hub(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, repo_name: str):
    """Push model and tokenizer to Hugging Face Hub."""
    # Make sure user is logged in
    login()  # will prompt for token if not already logged in

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model and tokenizer pushed to Hugging Face Hub repo: {repo_name}")

def load_model_from_dir(model_dir: str, num_labels: int = 2):
    """Load saved model and tokenizer from local directory."""
    from transformers import BertForSequenceClassification, BertTokenizer
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded model from {model_dir} to {device}")
    return model, tokenizer, device

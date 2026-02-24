import torch
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def format_time(elapsed: float) -> str:
    """Format seconds into hh:mm:ss string."""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def create_dataloader(inputs, masks, labels, batch_size=32, mode='train'):
    """Create PyTorch DataLoader for training/validation/testing."""
    tensor_inputs = torch.tensor(inputs)
    tensor_labels = torch.tensor(labels)
    tensor_masks = torch.tensor(masks)

    dataset = TensorDataset(tensor_inputs, tensor_masks, tensor_labels)

    if mode == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

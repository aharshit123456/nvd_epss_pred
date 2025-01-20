from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch


class NVDDataset(Dataset):
    def __init__(self, descriptions, cvss_scores, targets, tokenizer, max_length):
        self.descriptions = descriptions
        self.cvss_scores = cvss_scores
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        # Tokenize text description
        encoding = self.tokenizer(
            self.descriptions[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'cvss_scores': torch.tensor(self.cvss_scores[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }
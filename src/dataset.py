"""
This module defines a custom PyTorch Dataset class for 
Grammar Error Correction (GEC).
It loads sentence pairs from a CSV file 
and converts them into tokenized tensors suitable for 
training transformer-based sequence-to-sequence models.

"""
import pandas as pd
import torch

class GrammarDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data.iloc[idx]["incorrect"]
        tgt = self.data.iloc[idx]["correct"]

        source = self.tokenizer(
            src,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer(
            tgt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


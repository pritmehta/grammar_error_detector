"""
handles the training and evaluation of a Grammar Error Correction (GEC) 
model using Hugging Face's Trainer API. 
It loads the model, prepares datasets, 
defines training configurations, 
and saves the trained model and tokenizer.

"""
import torch
from transformers import Trainer, TrainingArguments
from dataset import GrammarDataset
from model import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = load_model()
model.to(device)

train_ds = GrammarDataset("../data/processed/train.csv", tokenizer)
val_ds = GrammarDataset("../data/processed/val.csv", tokenizer)

args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    # tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("../saved_model")
tokenizer.save_pretrained("../saved_model")



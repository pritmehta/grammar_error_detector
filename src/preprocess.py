"""
This script loads a large grammar correction dataset,
cleans it, splits it into training and validation sets,
and saves the processed data as CSV files.

"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "../data/raw/C4_200M_data.tsv"
SAMPLES = 200_000

print("Loading dataset...")
df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    names=["incorrect", "correct"],
    nrows=SAMPLES
)

df.dropna(inplace=True)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv("../data/processed/train.csv", index=False)
val_df.to_csv("../data/processed/val.csv", index=False)

print("Preprocessing complete")

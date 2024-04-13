import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import EvalPrediction
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import argparse  # Import argparse

class TextPairDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_pair = self.data[idx]["text"]
        label = self.data[idx]["label"]
        encoding = self.tokenizer(text_pair[0], text_pair[1], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        # Flatten the output dictionary tensors for Trainer compatibility
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label)
        return encoding
def DFtoData(df):
    # Process the CSV file to extract the relevant text and labels
    data = []
    for index, row in df.iterrows():
        text1 = row['sentence1']
        text2 = row['sentence2']
        label = row['label']
        data.append({"text": (text1, text2), "label": label})
    return data

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser(description='Train a RoBERTA model on text pair classification')
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model version to use (e.g., roberta-base, roberta-large)')
    args = parser.parse_args()

    # Load the CSV file
    train_df = pd.read_csv(r'data/train.csv') 
    test_df = pd.read_csv(r'data/test.csv')
    eval_df = pd.read_csv(r'data/eval.csv')

    train_data = DFtoData(train_df)
    test_data = DFtoData(test_df)
    eval_data = DFtoData(eval_df)
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Prepare datasets
    train_dataset = TextPairDataset(tokenizer, train_data)
    eval_dataset = TextPairDataset(tokenizer, eval_data)
    test_dataset = TextPairDataset(tokenizer, test_data)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory for checkpoints
        num_train_epochs=6,              # total number of training epochs
        per_device_train_batch_size=16,   # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        evaluation_strategy="epoch",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Optionally, evaluate on the test set after training
    print("Evaluating on test data...")
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

if __name__ == "__main__":
    main()
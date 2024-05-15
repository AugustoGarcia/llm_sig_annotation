import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader

datafile=['Hydroxychloroquine 200 mg.csv','Prednisone.csv']
datafile='Prednisone.csv'
models=['emilyalsentzer/Bio_ClinicalBERT',"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"]
model='bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'

file=os.path.join(os.getcwd(),'data', datafile)
df=pd.read_csv(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df=df[df['set_0']=='train']
test_df=df[df['set_0']=='test']

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# Custom Dataset class
class ClinicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        output = self.dataframe.iloc[idx]['output']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'output': torch.tensor(output, dtype=torch.float)
        }

# Initialize datasets and dataloaders
train_dataset = ClinicalDataset(train_df, tokenizer)
test_dataset = ClinicalDataset(test_df, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

# Model definition
class ClinicalPredictor(nn.Module):
    def __init__(self):
        super(ClinicalPredictor, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        return self.linear(cls_output).squeeze(-1)

# Initialize model, optimizer, and loss function
model = ClinicalPredictor()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Training and Testing loop
for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for batch in train_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = batch['output'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = batch['output'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, output)

            test_loss += loss.item()
        
        test_loss /= len(test_loader)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}")


# Evaluation loop
model.eval()
with torch.no_grad():
    test_predictions = []
    test_actuals = []
    for batch in test_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = batch['output'].to(device)

        predictions = model(input_ids, attention_mask)
        test_predictions.extend(predictions.cpu().tolist())
        test_actuals.extend(output.cpu().tolist())

# Print test results
print("Test Predictions:", test_predictions)
print("Test Actuals:", test_actuals)

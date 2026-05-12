import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

"""
What is being done:

Big Picture: We take raw data and make Transformer ready

We take the CSV row
We extract the text
Tokenizor converts the text, we also add CLS SEP and truncate and make tensor format
We make one essay as 1 item
Process data accurately for either training or test environment
"""

class EssayDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text']) # just pandas stuff to get our test essay
        
        # Tokenize sequence, automatically adding [CLS] and [SEP]
        encoding = self.tokenizer(
            text, # Test Essay
            add_special_tokens=True, # Add [CLS] and [SEP]
            max_length=self.max_length, # No more than 512 tokens default
            padding='max_length', # Make sure same length
            truncation=True, # If too long cut off
            return_tensors='pt' # Return Pytorch tensors
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if not self.is_test:
            # Map labels: positive (LLM) is 1.0 and negative (student) is 0.0
            label_str = self.df.iloc[idx]['label']
            label = 1.0 if label_str == 'positive' else 0.0
            item['labels'] = torch.tensor(label, dtype=torch.float)
        else:
            item['id'] = self.df.iloc[idx]['id']
            
        return item
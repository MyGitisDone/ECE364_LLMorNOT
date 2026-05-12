import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

"""
What is being done: We are using a manually Shrunk MiniLM to train our model

We take the configuration from the MiniLM model and we manually define parameters so total under 15M

"""

class ECE364Classifier(nn.Module):
    def __init__(self, model_name="nreimers/MiniLM-L6-H384-uncased"):
        super(ECE364Classifier, self).__init__()
        # Load configuration only (no weights)
        config = AutoConfig.from_pretrained(model_name)

        # Modify the config to guarantee we are under the 15M limit
        config.num_hidden_layers = 4  # Reduced from 6 to save parameters
        config.hidden_size = 256  # Reduced from 384
        config.num_attention_heads = 8
        
        self.encoder = AutoModel.from_config(config)
        
        # Classification Head
        self.dropout = nn.Dropout(0.2) # Randomly remove 20% of neurons so that we don't overfit
        self.classifier = nn.Linear(config.hidden_size, 1) 
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    
        # Extract the hidden state of the [CLS] token (always at index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return torch.sigmoid(logits) # Output a probability between 0 and 1 so using sigmoid
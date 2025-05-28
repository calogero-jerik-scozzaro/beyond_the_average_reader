import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BertWithLayer (nn.Module):
    def __init__(self,config):
        super(BertWithLayer, self).__init__()        
        bert_config = AutoConfig.from_pretrained(config["bert_model_name"])
        bert_config.output_hidden_states=True
        self.bert_layers = AutoModel.from_pretrained(config["bert_model_name"], config = bert_config)
        self.bert_layers.eval()
        if not config["fine_tuning"]:
            for param in self.bert_layers.parameters():
                param.requires_grad = False
                
        self.linear_layer = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, word_ids):
        output = self.bert_layers(input_ids=input_ids, attention_mask=attention_mask)
        hidden_rep = output[0]                  # [batch_size, seq_length, hidden_size]
        output = self.linear_layer(hidden_rep)  # [batch_size, seq_length, 1]
        return output

  


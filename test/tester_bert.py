import torch
import torch.nn as nn

class TesterBERT(nn.Module):
    def __init__(self, model, test_loader, config):
        super(TesterBERT, self).__init__()
        self.model = model
        self.test_loader = test_loader
        self.config = config

    def evaluate(self):
        self.model.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0
        predictions = []
        outputs = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.config['device'])
                attention_mask = batch['attention_mask'].to(self.config['device'])
                word_ids = batch['word_ids'].to(self.config['device'])
                output = batch['output'].to(self.config['device'])
                prediction = self.model(input_ids, attention_mask, word_ids).squeeze(-1)
                loss = criterion(prediction, output)
                total_loss += loss.item()

                predictions.extend(prediction.cpu().numpy())
                outputs.extend(output.cpu().numpy())

        return predictions, outputs
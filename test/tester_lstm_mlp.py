import torch
import torch.nn as nn

class TesterLSTMMLP(nn.Module):
    def __init__(self, model, test_loader, config):
        super(TesterLSTMMLP, self).__init__()
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
                features = batch['features'].to(self.config['device'])
                output = batch['output'].to(self.config['device'])
                pred = self.model(features).squeeze(-1)
                loss = criterion(pred, output)
                total_loss += loss.item()

                predictions.extend(pred.cpu().numpy())
                outputs.extend(output.cpu().numpy())

        return predictions, outputs
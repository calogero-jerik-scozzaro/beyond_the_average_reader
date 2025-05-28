import torch
import torch.nn as nn

class TesterLSTM(nn.Module):
    def __init__(self, model, test_loader, config):
        super(TesterLSTM, self).__init__()
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
                pred, (h_n, c_n) = self.model(features)
                prediction = h_n[-1]  # Get the last hidden state
                loss = criterion(prediction, output)
                total_loss += loss.item()

                predictions.extend(prediction.cpu().numpy())
                outputs.extend(output.cpu().numpy())

        return predictions, outputs

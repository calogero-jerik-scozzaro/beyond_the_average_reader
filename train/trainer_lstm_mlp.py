import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from utils import early_stopping

class TrainerLSTMMLP:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optim = AdamW(model.parameters(), lr=5e-3)
        self.scheduler = LinearLR(self.optim, start_factor=1.0, end_factor=0.0, total_iters=config['epochs'])
        self.criterion = torch.nn.MSELoss()
        self.model.to(config['device'])
        self.min_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None

    def train(self):
        self.model.train()
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            for batch in self.train_loader:
                features = batch['features'].to(self.config['device'])
                output = batch['output'].to(self.config['device'])
                mask = (output != -1).float()  # 1 for relevant tokens, 0 otherwise
                self.optim.zero_grad()
                prediction = self.model(features).squeeze(-1)
                masked_prediction = prediction * mask
                masked_output = output * mask
                loss = self.criterion(masked_prediction, masked_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                epoch_loss += loss.item()
            
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(self.config['device'])
                    output = batch['output'].to(self.config['device'])
                    mask = (output != -1).float()
                    prediction = self.model(features).squeeze(-1)
                    masked_prediction = prediction * mask
                    masked_output = output * mask
                    loss = self.criterion(masked_prediction, masked_output)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)
            # Early stopping check
            stop, self.min_loss, self.best_model_state, self.epochs_no_improve = early_stopping(self.model, val_loss, self.min_loss, self.epochs_no_improve)
            if stop:
                break

            self.model.train()
            self.scheduler.step()
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model

    def fine_tune(self, train_loader_ft, model):
        self.model = model
        self.train_loader = train_loader_ft
        self.optim = AdamW(self.model.parameters(), lr=5e-4)
        self.scheduler = LinearLR(self.optim, start_factor=1.0, end_factor=0.0, total_iters=self.config['ft_epochs'])
        self.model.to(self.config['device'])
        self.min_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = self.model.state_dict().copy()
        self.model.train()
        for epoch in range(self.config['ft_epochs']):
            epoch_loss = 0
            for batch in self.train_loader:
                features = batch['features'].to(self.config['device'])
                output = batch['output'].to(self.config['device'])
                mask = (output != -1).float()
                self.optim.zero_grad()
                prediction = self.model(features).squeeze(-1)
                masked_prediction = prediction * mask
                masked_output = output * mask
                loss = self.criterion(masked_prediction, masked_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            self.scheduler.step()
            stop, self.min_loss, self.best_model_state, self.epochs_no_improve = early_stopping(self.model, epoch_loss, self.min_loss, self.epochs_no_improve)
            if stop:
                break
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.model


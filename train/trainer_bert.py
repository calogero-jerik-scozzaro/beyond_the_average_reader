import torch
from utils import early_stopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

class TrainerBERT:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optim = AdamW(model.parameters(), lr=5e-5)
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
                input_ids = batch['input_ids'].to(self.config['device'])
                attention_mask = batch['attention_mask'].to(self.config['device'])
                word_ids = batch['word_ids'].to(self.config['device'])
                output = batch['output'].to(self.config['device'])
                mask = (output != -1).float()  # 1 for relevant tokens, 0 otherwise
                self.optim.zero_grad()
                prediction = self.model(input_ids, attention_mask, word_ids).squeeze(-1)
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
                    input_ids = batch['input_ids'].to(self.config['device'])
                    attention_mask = batch['attention_mask'].to(self.config['device'])
                    word_ids = batch['word_ids'].to(self.config['device'])
                    output = batch['output'].to(self.config['device'])
                    mask = (output != -1).float()
                    prediction = self.model(input_ids, attention_mask, word_ids).squeeze(-1)
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

'''    
def fine_tuning_bert(model, train_loader, config):
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    total_steps = len(train_loader) * config['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: 1 - step / total_steps)

    criterion = torch.nn.MSELoss()
    model.train()
    model.to(config['device'])
    config['epochs'] = 1000
    # train the model all_without_reader
    min_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(config['epochs']):
        #print(f"Epoch {epoch+1}/{config['epochs']}")
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            word_ids = batch['word_ids'].to(config['device'])
            output = batch['output'].to(config['device'])
            mask = (output != -1).float()  # 1 for relevant tokens, 0 otherwise

            optim.zero_grad()
            prediction = model(input_ids, attention_mask, word_ids).squeeze(-1)
            
            masked_prediction = prediction * mask
            masked_output = output * mask

            loss = criterion(masked_prediction, masked_output)            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        stop, min_loss, best_model_state, epochs_no_improve = early_stopping(model, epoch_loss, min_loss, epochs_no_improve)

        if stop:
           break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
   
    return model
'''
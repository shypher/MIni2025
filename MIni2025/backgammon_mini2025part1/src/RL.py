
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR


path= "c:/Users/shay1/Documents/GitHub/MIni2025/MIni2025/backgammon_mini2025part1/RL_database.npy"
class BackgammonNet(nn.Module):  
    def __init__(self, input_size=29):
        super(BackgammonNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 128) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(64, 64)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(64, 32)
        self.dropout6 = nn.Dropout(0.1)
        self.output = nn.Linear(32, 1)
        self.relu6 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()   
        self.relu7 = nn.ReLU()
        self.elu = nn.ELU()
        self.oldoutput = nn.Linear(32,1)
        self.FineTunefc = nn.Linear(32, 32)
        self.FineTunerelu = nn.ReLU()
        for name, param in self.named_parameters():
            if ("FineTunefc" not in name and "FineTunerelu" not in name and "output" not in name):
                param.requires_grad = False


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x) 
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x) 
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x) 
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x) 
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x) 
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x) 
        x = self.relu7(x)
        #fine tuning layer
        x= self.FineTunefc(x) 
        x= self.FineTunerelu(x) 
        x= self.output(x)
        return self.sigmoid(x)  # Final heuristic score


    @staticmethod
    def train_network(dataset_path=path, batch_size=1024, num_epochs=4000, learning_rate=0.001, input_size=29):  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        dataset = np.load(dataset_path, allow_pickle=True)
        if isinstance(dataset[0], dict):
            X = np.array([np.concatenate([entry['board'], [entry['color']]]) for entry in dataset], dtype=np.float32)  # Combine board and color
            y = np.array([entry['RL_score'] for entry in dataset], dtype=np.float32)  # Heuristic score is the target

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Features
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)  
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        model = BackgammonNet(input_size=input_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()
        model.train()
        patience = 20  
        best_val_loss = float('inf')
        counter = 0
        for epoch in range(num_epochs):
            
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter=0
            else:
                counter += 1
            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Learning Rate: {current_lr}")
            if counter>= patience:
                print(f"Early stopping at epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                break
            
            if(epoch+1==num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            model.train()


        torch.save(model.state_dict(), "RLNN_lvl2.pth")
        print("Training complete. Model saved to RLNN_lvl2.pth")

        return model, test_set
    @staticmethod
    def evaluate_test_set(model, test_set):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Set Loss: {avg_test_loss:.6f}")

    
        
if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    model, test_set = BackgammonNet.train_network()
    BackgammonNet.evaluate_test_set(model, test_set)
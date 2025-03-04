
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


path= "c:/Users/shay1/Documents/GitHub/MIni2025/MIni2025/backgammon_mini2025part1/normalized_database.npy"
class BackgammonNet(nn.Module):  
    def __init__(self, input_size=29):
        super(BackgammonNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        #self.nr1 = nn.BatchNorm1d(40)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 128) 
        #self.nr2 = nn.BatchNorm1d(40)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 128)
        #self.nr3 = nn.BatchNorm1d(40)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 64)
        #self.nr4 = nn.BatchNorm1d(40)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(64, 64)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(64, 32)
        self.dropout6 = nn.Dropout(0.1)
        #self.dropout4 = nn.Dropout(0.1)
        #self.fc5 = nn.Linear(40, 40)
        #self.nr5 = nn.BatchNorm1d(40)
        #self.relu5 = nn.ReLU()
        self.output = nn.Linear(32, 1)
        #self.dropout5 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()    
        self.elu = nn.ELU()


    def forward(self, x):
        #x = x * 100  # הגדלת הערכים

        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout1(x) 
        x = self.fc2(x)
        x = self.elu(x)
        x = self.dropout2(x) 
        x = self.fc3(x)
        x = self.elu(x)
        x = self.dropout3(x) 
        x = self.fc4(x)
        x = self.elu(x)
        x = self.dropout4(x) 
        x = self.fc5(x)
        x = self.elu(x)
        x = self.dropout5(x) 
        x = self.fc6(x)
        x = self.elu(x)
        x = self.dropout6(x) 
        x = self.output(x)
        #x= x/100
        #x = torch.tanh(x)  # ערכים בין -1 ל-1
        #x = (x + 1) / 2  # שינוי לטווח [0,1]
        return self.sigmoid(x)  # Final heuristic score
    """def init(self, input_size=29):
        super(BackgammonNet, self).init()
        self.fc1 = nn.Linear(input_size, 40)
        self.nr1 = nn.BatchNorm1d(40)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(40, 40)
        self.nr2 = nn.BatchNorm1d(40)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(40, 40)
        self.nr3 = nn.BatchNorm1d(40)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)

        self.output = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nr1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.nr2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.nr3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.output(x)
        return x  # Final heuristic score"""

    @staticmethod
    def train_network(dataset_path=path, batch_size=1024, num_epochs=2000, learning_rate=2e-4, input_size=29):  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        dataset = np.load(dataset_path, allow_pickle=True)  # Load dataset
        if isinstance(dataset[0], dict):
            X = np.array([np.concatenate([entry['board'], [entry['color']]]) for entry in dataset], dtype=np.float32)  # Combine board and color
            y = np.array([entry['RL_score'] for entry in dataset], dtype=np.float32)  # Heuristic score is the target

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Features
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)  
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create the network and optimizer
        model = BackgammonNet(input_size=input_size).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-4)
        criterion = nn.MSELoss()
        model.train()
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
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.device_count())  # Should return the number of GPUs
    print(torch.cuda.get_device_name(0))  # Should return "NVIDIA GeForce RTX 4070 Ti"
    model, test_set = BackgammonNet.train_network()
    BackgammonNet.evaluate_test_set(model, test_set)
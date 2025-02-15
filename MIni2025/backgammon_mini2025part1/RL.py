from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import threading
from itertools import permutations
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.strategy_factory import StrategyFactory
class BackgammonNet(nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(40, 1)
        self.sigmoid = nn.Sigmoid() 
        self.softmax = nn.Softmax(dim=1)
    
    @staticmethod
    def train_network(dataset_path="c:/Users/shay1/Documents/GitHub/MIni2025/MIni2025/backgammon_mini2025part1/normalized_database.npy", batch_size=32, num_epochs=5, learning_rate=1e-3, input_size=48):   
        dataset = np.load(dataset_path, allow_pickle=True)  # Load dataset
        print("Dataset dtype:", dataset.dtype)  # Debugging print

        # Ensure dataset contains only numerical values
        if isinstance(dataset[0], dict):
            dataset = np.array([[entry['color'], entry['heuristic_score']] + entry['board'] for entry in dataset], dtype=np.float32)
        dataset = torch.from_numpy(dataset).float()  # Convert to PyTorch tensor

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create the network and optimizer
        model = BackgammonNet(input_size=input_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
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
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            model.train()


        torch.save(model.state_dict(), "RLNN.pth")
        print("Training complete. Model saved to RLNN.pth")

        return model, test_set
    @staticmethod
    def evaluate_test_set(model, test_set):
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Set Loss: {avg_test_loss:.6f}")
        
if __name__ == '__main__':
    model, test_set = BackgammonNet.train_network()
    BackgammonNet.evaluate_test_set(model, test_set)
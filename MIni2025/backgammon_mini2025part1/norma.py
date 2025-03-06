from matplotlib import pyplot as plt
import torch
from src.colour import Colour
from random import randint
from scipy.special import erf
from src.RL_old import BackgammonNet 
from random import randint
import numpy as np
import torch.nn as nn
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BackgammonNet().to(device)
    try:
        model.load_state_dict(torch.load("RLNN_finalLvlA.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
    norm_db = np.load('normalized_database.npy', allow_pickle=True )
    differences = [] 
    for i in range(0, 200):
        entry = norm_db[i]
        color, score, board = entry['color'], entry['heuristic_score'], entry['board']
        input_data = np.concatenate([board, np.array([(color+1)%2], dtype=np.float32)])
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        input_tensor = input_tensor.to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(input_tensor).item()
        diff = output - score
        differences.append(diff)
    differences = np.array(differences)
    mean_diff = np.mean(differences)
    variance_diff = np.var(differences)
    total_diff = np.sum(differences)
    print(f"Average:{mean_diff:.6f}")
    print(f"Difference Variance: {variance_diff:.6f}")
    print(f"Difference sum: {total_diff:.6f}")
        
        
        
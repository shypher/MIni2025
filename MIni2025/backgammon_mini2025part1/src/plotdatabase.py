import torch
import numpy as np
import matplotlib.pyplot as plt
from RL_old import BackgammonNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BackgammonNet().to(device)
    model.load_state_dict(torch.load("RLNN_new.pth", map_location=device))
    model.eval()

    # 2️⃣ טען את בסיס הנתונים
    dataset = np.load("normalized_database.npy", allow_pickle=True)
    model.load_state_dict(torch.load("RLNN_new.pth", map_location=device), strict=False)
    # בחר 500 דוגמאות לבדיקה
    X_test = np.array([np.concatenate([entry['board'], [entry['color']]]) for entry in dataset[:500]], dtype=np.float32)
    # Load the database
    # המר לטנסור של PyTorch
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    # Select 500 examples for testing
    # 3️⃣ קבל תחזיות מהרשת
    with torch.no_grad():
    # Convert to PyTorch tensor
        predictions = model(X_tensor).cpu().numpy().flatten()
    # 4️⃣ צייר היסטוגרמה של התחזיות
    # Get predictions from the network
    plt.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Predicted Heuristic Value")
    plt.ylabel("Frequency")
    # Draw a histogram of the predictions
    plt.grid(True)

    # הצג את ההיסטוגרמה
    plt.show()    # Display the histogram

# Final Project – Reinforcement Learning in Backgammon

**Submission Date:** 06/03/2025


---

## Overview

This project develops a Backgammon player using multiple approaches. Our system includes heuristic, MCTS, Minimax players, and an RL-based player. The reinforcement learning (RL) player is trained to predict a normalized heuristic value (between 0 and 1) for board states and is then used for move selection. 

---

## RL Structure

### RL_database
- **Board_Generator & set_database**  
  Simulate games using different strategies (heuristic-based and random moves) to collect board states and score values.  

### RL (the main neural network)
- **Input:** Vector containing board state and player color.
- **Architecture:** 6 fully-connected layers (3 layers of 128 neurons, 2 layers of 64, one layer reducing to 32), and an additional 32 sized fine tune layer.
- **Output:** A single neuron (passed through sigmoid) predicts the heuristic score.
- **Training:** Utilizes Adam optimizer with a StepLR scheduler and Early Stopping; most original layers are frozen during fine tuning.

### RL_Player
- **RL Player:** Uses the trained neural network to evaluate the possible moves and select the best move.

### RL_database_old
- **Board_Generator & set_database**  
  The data base of the first part of the RL. Where we simulate games, break them down to boards and match them with a noramlized heuristic value using the heuristic player.  

### RL_old
- Achived extrimly good results (loss: 0.00012%)
- **Input:** Vector containing board state and player color.
- **Architecture:** 6 fully-connected layers (3 layers of 128 neurons, 2 layers of 64, one layer reducing to 32).
- **Output:** A single neuron (passed through sigmoid) predicts the heuristic score.
- **Training:** Utilizes Adam optimizer with a StepLR scheduler and Early Stopping.

### RL_Player_old
- **RL Player_old:** Uses the step 1 trained neural network to evaluate the possible moves and select the best move.
- - We've kept it for documentation of the process and to use it whenever we want to compare for our step 1 model.
  
### Other Classes
- **Other class:** The project also includes heuristic, MCTS, and Minimax players which won't be covered in this readme file.

---

## System Requirements

- Python 3 (if your system runs Python 3 using the command `python3`, you may need to change the commands below accordingly)

---

## How to Run

- **Human vs Computer:**  
  Run `python single_player.py`, then choose the computer strategy to play against.

- **Human vs Human:**  
  Run `python two_player.py`.

- **Computer vs Computer:**  
  Run `python main.py`.  
  (The two "players" can have different strategies.)

- **Tournament:**  
  Run `python tournament.py`.

---

## Notes

- The project includes various player implementations. RL is one of the strategies available.


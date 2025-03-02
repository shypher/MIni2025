import os
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import torch
from src.colour import Colour
from src.compare_all_moves_strategy import CompareAllMovesWeightingDistanceAndSinglesWithEndGame
from src.game2 import Game
from src.strategies import MoveRandomPiece

from src.random_comper import CompareAllMovesWeightingDistanceAndSinglesWithEndGame_random, CompareAllMovesSimple_random
import numpy as np 
from random import randint
from scipy.special import erf
from src.RL import BackgammonNet
from src.RL_player import RL_player, RL_player_random



  
    
def Board_Generator(size=1000): 
    database = []
    for i in range(size):
        game = Game(
            white_strategy= MoveRandomPiece(),
            black_strategy= MoveRandomPiece(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
        if i == 0:
            database = np.concatenate((database, game.get_game_history()))
        elif i > 100:
            
            game_history = game.get_game_history()[5:-5]  
            database = np.concatenate((database, game_history))
        else:
            #remove the first element of the game history
            game_history = game.get_game_history()[1:] 
            database = np.concatenate((database, game_history))
        print("game ", i+1, " done")
        
    size = database.nbytes

    #play a random game
    print("database size: ", len(database))
    print("database bytes size: ", size)
    #save the database
    np.save('database.npy', database, allow_pickle=True) 
def Board_Generator_RL(size=128): 
    database = []
    all_dist=[]
    for i in range(size):
        game = Game(
            white_strategy= RL_player_random(),
            black_strategy= RL_player_random(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
        new_db = []
        winner = game.who_won()
        board_history = game.get_game_history()
        lastBoard = board_history[-1]['board']
        dis_sum = 0 
        lastBoardLoc = lastBoard[:-4]
        eat_black= lastBoard[25]
        eat_white= lastBoard[24]
        indices = np.arange(len(lastBoardLoc))
        distances = np.abs(indices - (24 if winner != Colour.WHITE else -1))
        dis_sum = np.sum(distances * np.abs(lastBoardLoc))
        dis_sum=25* (eat_black+eat_white)+dis_sum
        
        max_dist = 105
            
        normalized_distance = min(dis_sum / max_dist, 1.0)    
        for state in board_history:
            Pcolor = state['color']
            Hboard = state['board']
            if winner.value == Pcolor:
                score = 0.6 + 0.4 * (normalized_distance)
            else:
                score = 0.4 * (1-normalized_distance)
            stateScoreInfo = {'color': Pcolor, 'RL_score': score, 'board': Hboard}
            new_db.append(stateScoreInfo)
            #log(f"color:{Pcolor}, board{lastBoard}\n score:{score}")
        database.extend(new_db)
        all_dist = np.append(all_dist, dis_sum)
        print("game ", i+1, " done")
        
    #size = database.nbytes

    #play a random game
    print("database size: ", len(database))
    #print("database bytes size: ", size)
    #save the database
    np.save('database.npy', database, allow_pickle=True)
    
    return all_dist

def set_RLdatabase():
    all_dist = Board_Generator_RL(size=100)  # Generates new board states
        
    # Load the RL-based database
    database = np.load('database.npy', allow_pickle=True)
    
    # Extract RL scores (already normalized)
    RL_scores = [entry['RL_score'] for entry in database]

    # Print RL score statistics
    print("Min RL score: ", np.min(all_dist))
    print("Max RL score: ", np.max(all_dist))
    print("Average RL score: ", np.mean(all_dist))
    print("Variance of RL score: ", np.var(all_dist))
    print("Standard deviation of RL score: ", np.std(all_dist))
    #std_all_dist = np.std(all_dist)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #heuristics= np.array([entry['RL_score'] for entry in database], dtype=np.float32).reshape(-1, 1)
    #qt = QuantileTransformer(output_distribution='uniform', random_state=0)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #normalized_RL = [0 if entry['RL_score'] == 0 else max(0.6,0.6 + 0.4 *(min(1, erf((entry['RL_score'] - 22.72) / (21.99 * 1.414))))) for entry in database]
    #for i, entry in enumerate(database):
        #entry['RL_score'] = normalized_RL[i]

    # Shuffle the database to introduce randomness
    np.random.shuffle(database)
    np.save('RL_database.npy', database, allow_pickle=True)
    # Print some random RL-based samples for logging
    RL_database = np.load('RL_database.npy', allow_pickle=True)


    

def log(message, file_path="db.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)   

def set_database():
    Board_Generator_RL(size = 5500)#
    #get the min and max values of heuristic in the database, the average value of the heuristic, variance, and standard deviation
    database = np.load('database.npy', allow_pickle=True)
    heuristics= [entry['heuristic_score'] for entry in database]
    
    print("min value of heuristic: ", np.min(heuristics))
    print("max value of heuristic: ", np.max(heuristics))
    print("average value of heuristic: ", np.mean(heuristics))
    print("variance of heuristic: ", np.var(heuristics))
    print("standard deviation of heuristic: ", np.std(heuristics))
    mean_heuristic = np.mean(heuristics)
    std_heuristic = np.std(heuristics)
    normalized_heuristics = [0.5 * (1.0 + erf((entry['heuristic_score'] - mean_heuristic) / (std_heuristic * np.sqrt(2.0)))) for entry in database]
    # Update the database with normalized heuristic scores
    for i, entry in enumerate(database):
        entry['heuristic_score'] = normalized_heuristics[i]
    #shffle the database
    np.random.shuffle(database)
    np.save('normalized_database.npy', database, allow_pickle=True)
    #print the normalized heuristic information
    print("min value of normalized heuristic: ", np.min(normalized_heuristics))
    print("max value of normalized heuristic: ", np.max(normalized_heuristics))
    print("average value of normalized heuristic: ", np.mean(normalized_heuristics))
    print("variance of normalized heuristic: ", np.var(normalized_heuristics))
    print("standard deviation of normalized heuristic: ", np.std(normalized_heuristics))
    #log 10 random objects from the normalized_database
    normalized_heuristics_db = np.load('normalized_database.npy', allow_pickle=True)
    database = np.load('database.npy', allow_pickle=True)
    for i in range(10):
        log("Normalized heuristic score: " + str(normalized_heuristics_db[i]['heuristic_score']))
stoper_path = r"C:\Users\shay1\Documents\GitHub\MIni2025\stoper.txt"
RL_database_path = r"C:\Users\shay1\Documents\GitHub\MIni2025\RL_database.npy"


RESULTS_PATH = "training_results.csv"

def save_training_result(iteration, average_score):
    file_exists = os.path.isfile(RESULTS_PATH)

    with open(RESULTS_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["iteration", "Average_Score"])
        writer.writerow([iteration, average_score])

def load_training_results():
    if not os.path.isfile(RESULTS_PATH):
        return [], []
    
    epochs, scores = [], []
    with open(RESULTS_PATH, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            epochs.append(int(row[0]))
            scores.append(float(row[1]))
    return epochs, scores

def plot_training_progress():
    epochs, scores = load_training_results()
    if not epochs:
        print("no data")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, scores, marker='o', linestyle='-')
    plt.xlabel("Training Steps")
    plt.ylabel("Average Score")
    plt.title("Training Progress of RL-Based Player")
    plt.grid(True)
    plt.savefig("training_progress.png") 
    plt.show()
    
TRAINING_STEP_PATH = "training_step.txt"
def load_training_step():
    """Loads the last training step from file, or starts at 1 if the file does not exist."""
    if os.path.isfile(TRAINING_STEP_PATH):
        with open(TRAINING_STEP_PATH, "r") as file:
            return int(file.read().strip())
    return 1  # Start from 1 if file doesn't exist
def save_training_step(step):
    """Saves the current training step to file."""
    with open(TRAINING_STEP_PATH, "w") as file:
        file.write(str(step))
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RL = BackgammonNet().to(device)
    training_step = load_training_step()
    
    while os.path.getsize(stoper_path) != 0:
        print("Run: ", training_step)
        set_RLdatabase()
        RL.train_network(dataset_path=RL_database_path)
        np.save(RL_database_path, [])
        total_score = 0
        num_games = 10
        for _ in range(num_games):
            game = Game(
                white_strategy= RL_player(),
                black_strategy= CompareAllMovesWeightingDistanceAndSinglesWithEndGame(),
                first_player=Colour(randint(0, 1)),
                time_limit=-1
            )
            game.run_game(verbose=False)          
            winner = game.who_won()
            lastBoard = game.get_game_history()
            lastBoard = lastBoard[-1]['board']
            print(f"last board: {lastBoard}")
            dis_sum = 0 
            lastBoardLoc = lastBoard[:-4]
            eat_black= lastBoard[25]
            eat_white= lastBoard[24]
            indices = np.arange(len(lastBoardLoc))
            distances = np.abs(indices - (24 if winner != Colour.WHITE else -1))
            dis_sum = np.sum(distances * np.abs(lastBoardLoc))
            dis_sum=25* (eat_black+eat_white)+dis_sum
            max_dist = 105
            
            normalized_distance = min(dis_sum / max_dist, 1.0)   
            if winner == Colour.WHITE:
                normalized_score =  0.6 + 0.4 * (1 - normalized_distance)
            else:
                normalized_score = 0.4 * (normalized_distance)
            print(f"Game {_ + 1} winner: {winner}, normalized score: {normalized_score}")
            total_score += normalized_score
        average_score = total_score / num_games
        save_training_result(training_step, average_score)
        training_step += 1
        save_training_step(training_step)

    plot_training_progress()

    
    




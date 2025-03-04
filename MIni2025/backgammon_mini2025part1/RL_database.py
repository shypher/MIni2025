import os
import csv
import concurrent
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import torch 
from tqdm import tqdm
from src.colour import Colour
from src.compare_all_moves_strategy import CompareAllMovesWeightingDistanceAndSinglesWithEndGame
from src.game2 import Game
from src.strategies import MoveRandomPiece
from src.random_comper import CompareAllMovesWeightingDistanceAndSinglesWithEndGame_random, CompareAllMovesSimple_random
import numpy as np 
from random import randint
from scipy.special import erf
from src.RL import BackgammonNet
from src.RL_player import RL_player, RL_player_new_model, RL_player_random

def Board_Generator_game(index):
    if (index%2==0):
        game = Game(
            white_strategy= RL_player_random(),
            black_strategy= RL_player_random(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
        winner = game.who_won()
        board_history = game.get_game_history()
        lastBoard = board_history[-1]['board']
        dis_sum = 0
        lastBoardLoc = lastBoard[:-4]
        eat_black = lastBoard[25]
        eat_white = lastBoard[24]
        indices = np.arange(len(lastBoardLoc))
        distances = np.abs(indices - (24 if winner != Colour.WHITE else -1))
        dis_sum = np.sum(distances * np.abs(lastBoardLoc))
        dis_sum = 25 * (eat_black + eat_white) + dis_sum
        max_dist = 105

        normalized_distance = min(dis_sum / max_dist, 1.0)
        new_db = []
        for state in board_history:
            Pcolor = state['color']
            Hboard = state['board']
            if winner.value == Pcolor:
                score = 0.6 + 0.4 * (normalized_distance)
            else:
                score = 0.3 * (1 - normalized_distance)
            stateScoreInfo = {'color': Pcolor, 'RL_score': score, 'board': Hboard}
            new_db.append(stateScoreInfo)

        return new_db, dis_sum
    else:
        game = Game(
            white_strategy= CompareAllMovesWeightingDistanceAndSinglesWithEndGame_random(),
            black_strategy= CompareAllMovesWeightingDistanceAndSinglesWithEndGame_random() ,
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
        winner = game.who_won()
        board_history = game.get_game_history()
        lastBoard = board_history[-1]['board']
        dis_sum = 0
        lastBoardLoc = lastBoard[:-4]
        eat_black = lastBoard[25]
        eat_white = lastBoard[24]
        indices = np.arange(len(lastBoardLoc))
        distances = np.abs(indices - (24 if winner != Colour.WHITE else -1))
        dis_sum = np.sum(distances * np.abs(lastBoardLoc))
        dis_sum = 25 * (eat_black + eat_white) + dis_sum
        max_dist = 105

        normalized_distance = min(dis_sum / max_dist, 1.0)
        new_db = []
        for state in board_history:
            Pcolor = state['color']
            Hboard = state['board']
            if winner.value == Pcolor:
                score = 0.6 + 0.4 * (normalized_distance)
            else:
                score = 0.3 * (1 - normalized_distance)
            stateScoreInfo = {'color': Pcolor, 'RL_score': score, 'board': Hboard}
            new_db.append(stateScoreInfo)

        return new_db, dis_sum

def Board_Generator_RL(size=128): 
    database = []
    all_dist = []
    progress_bar = tqdm(total=size, desc="Generating Boards", position=0, leave=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(Board_Generator_game, i): i for i in range(size)}

        for future in concurrent.futures.as_completed(futures):
            game_data, dis_sum = future.result()
            database.extend(game_data)
            all_dist.append(dis_sum)
            progress_bar.update(1) 

    progress_bar.close()
    print("\n Board Generation Completed!")
    np.save('database.npy', database, allow_pickle=True)   
    """database = []
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
    
    return all_dist"""
    

def set_RLdatabase():
    Board_Generator_RL(size=200)  # Generates new board states
        
    # Load the RL-based database
    database = np.load('database.npy', allow_pickle=True)
    
    # Extract RL scores (already normalized)
    RL_scores = [entry['RL_score'] for entry in database]

    # Print RL score statistics
    """print("Min RL score: ", np.min(all_dist))
    print("Max RL score: ", np.max(all_dist))
    print("Average RL score: ", np.mean(all_dist))
    print("Variance of RL score: ", np.var(all_dist))
    print("Standard deviation of RL score: ", np.std(all_dist))"""
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
def evaluate_against_old_model(num_games=10):
    new_wins = 0
    old_wins = 0

    with tqdm(total=num_games, desc="Evaluating Models", position=0, leave=True) as pbar:
        for _ in range(num_games):
            game = Game(
                white_strategy=RL_player(),
                black_strategy=RL_player_new_model(),
                first_player=Colour(randint(0, 1)),
                time_limit=-1
            )
            game.run_game(verbose=False)
            if game.who_won() == Colour.WHITE:
                new_wins += 1
            else:
                old_wins += 1
            pbar.update(1)

    print(f"\nResults: New Model {new_wins} - {old_wins} Old Model")
    return new_wins, old_wins 
       
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RL = BackgammonNet().to(device)
    training_step = load_training_step()
    
    while os.path.getsize(stoper_path) != 0:
        print("Run: ", training_step)
        set_RLdatabase()
        RL.train_network(dataset_path=RL_database_path)
        new_wins, old_wins = evaluate_against_old_model()
        if new_wins > old_wins and new_wins / (new_wins + old_wins) >= 0.55:
            torch.save(RL.state_dict(), "RLNN_lvl2_best.pth")
            print("Save new DB")
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
                normalized_score =  0.6 + 0.4 * (normalized_distance)
            else:
                normalized_score = 0.4 * (1 - normalized_distance)
            print(f"Game {_ + 1} winner: {winner}, normalized score: {normalized_score}")
            total_score += normalized_score
        average_score = total_score / num_games
        save_training_result(training_step, average_score)
        training_step += 1
        save_training_step(training_step)

    plot_training_progress()

    
    




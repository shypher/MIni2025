from matplotlib import pyplot as plt
from src.colour import Colour
from src.compare_all_moves_strategy import CompareAllMovesWeightingDistanceAndSinglesWithEndGame
from src.game2 import Game
from src.strategies import MoveRandomPiece
from src.RL_player import RL_player
import numpy as np 
from random import randint
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from src.RL import BackgammonNet

  
    
def Board_Generator(size=5500): 
    database = []
    for i in range(int(size/2)):
        game = Game(
            white_strategy= CompareAllMovesWeightingDistanceAndSinglesWithEndGame(),
            black_strategy= CompareAllMovesWeightingDistanceAndSinglesWithEndGame(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
            #remove the first element of the game history
        game_history = game.get_game_history()[1:] 
        database = np.concatenate((database, game_history))
        print("game ", i+1, " done")
        game = Game(
            white_strategy= MoveRandomPiece(),
            black_strategy= MoveRandomPiece(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
        game_history = game.get_game_history()[1:] 
        database = np.concatenate((database, game_history))
        print("game ", i+1, " done")
        
        
    size = database.nbytes

    #play a random game
    print("database size: ", len(database))
    print("database bytes size: ", size)
    #save the database
    np.save('database.npy', database, allow_pickle=True) 
    print("database size: ", len(database))
    print("database bytes size: ", size)
    np.save('database.npy', database, allow_pickle=True)


    

def log(message, file_path="db.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)   

def set_database():
    Board_Generator(size = 5500)
    #get the min and max values of heuristic in the database, the average value of the heuristic, variance, and standard deviation
    
    database = np.load('database.npy', allow_pickle=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    heuristics= np.array([entry['heuristic_score'] for entry in database], dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_heuristics = scaler.fit_transform(heuristics).flatten()

    plt.hist(heuristics, bins=50, alpha=0.5, label="Original Heuristic Distribution")
    plt.legend()
    plt.show()
    print("min value of heuristic: ", np.min(heuristics))
    print("max value of heuristic: ", np.max(heuristics))
    print("average value of heuristic: ", np.mean(heuristics))
    print("variance of heuristic: ", np.var(heuristics))
    print("standard deviation of heuristic: ", np.std(heuristics))
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
    plt.hist(normalized_heuristics, bins=50, alpha=0.5, label="Normalized Heuristic Distribution")
    plt.legend()
    plt.show()
    normalized_heuristics_db = np.load('normalized_database.npy', allow_pickle=True)
    database = np.load('database.npy', allow_pickle=True)
    

    
if __name__ == '__main__':
    set_database()
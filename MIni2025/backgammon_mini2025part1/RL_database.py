from src.colour import Colour
from src.compare_all_moves_strategy import CompareAllMovesWeightingDistanceAndSinglesWithEndGame
from src.game2 import Game
from src.strategies import MoveRandomPiece
import numpy as np 
from random import randint
from scipy.special import erf



  
    
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

def log(message, file_path="db.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)   

def set_database():
    Board_Generator(size = 5500)
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
    
if __name__ == '__main__':
    set_database()

    
    




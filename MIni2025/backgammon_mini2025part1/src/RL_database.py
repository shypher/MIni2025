from src.colour import Colour
from src.compare_all_moves_strategy import CompareAllMovesWeightingDistanceAndSinglesWithEndGame
from src.game2 import Game
from src.strategies import MoveRandomPiece
from src.RL_player import RL_player
import numpy as np 
from random import randint
from scipy.special import erf
from MIni2025.backgammon_mini2025part1.src.RL import BackgammonNet


  
    
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
def Board_Generator_RL(size=1000): 
    database = []
    for i in range(size):
        game = Game(
            white_strategy= RL_player(),
            black_strategy= RL_player(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        new_db = np.array([])
        winner = game.who_won()
        board_history = game.get_game_history()
        lastBoard = board_history[-1]['board']
        log("last board" + str(lastBoard))
        black_dis_sum = 0 
        white_dis_sum = 0
        for loc in lastBoard:
            if winner == Colour.BLACK:
                white_dis_sum = loc*abs(lastBoard[loc])
                #TODO: get the mean and std of the distance sum, for now we will use 7 and 3
            else #winner == Colour.WHITE:
                black_dis_sum = loc*abs(24-lastBoard[loc])
                "0.6-1"
        if winner == Colour.BLACK:
            black_dis_sum = 0.6 + 0.4 * (1 + erf((black_dis_sum - mean_black) / (std_black * np.sqrt(2)))) / 2
            white_dis_sum = 0 + 0.4 * (1 + erf((black_dis_sum - mean_black) / (std_black * np.sqrt(2)))) / 2
        else:
            white_dis_sum = 0.6 + 0.4 * (1 + erf((black_dis_sum - mean_black) / (std_black * np.sqrt(2)))) / 2
            black_dis_sum = 0 + 0.4 * (1 + erf((black_dis_sum - mean_black) / (std_black * np.sqrt(2)))) / 2



            
            
        for Pcolor, score, Hboard in board_history:
            if winner != Pcolor:
                score = 0
                #score(true,Pcolor, lastBoard)
            else:
                score=1  

              
            stateScoreInfo = {'color': Pcolor, 'RL_Score': score, 'board':Hboard}
            new_db.append()       
        database = np.concatenate((database, ))
        
    size = database.nbytes

    #play a random game
    print("database size: ", len(database))
    print("database bytes size: ", size)
    #save the database
    np.save('database.npy', database, allow_pickle=True)

def set_RLdatabase():
    for i in range(1, 5001): 
        Board_Generator_RL(size=1)  # Generates new board states
        
        # Load the RL-based database
        database = np.load('database.npy', allow_pickle=True)
        
        # Extract RL scores (already normalized)
        RL_scores = [entry['RL_score'] for entry in database]

        # Print RL score statistics
        print("Min RL score: ", np.min(RL_scores))
        print("Max RL score: ", np.max(RL_scores))
        print("Average RL score: ", np.mean(RL_scores))
        print("Variance of RL score: ", np.var(RL_scores))
        print("Standard deviation of RL score: ", np.std(RL_scores))

        # Shuffle the database to introduce randomness
        np.random.shuffle(database)
        np.save('RL_database.npy', database, allow_pickle=True)

        # Print some random RL-based samples for logging
        RL_database = np.load('RL_database.npy', allow_pickle=True)
        for i in range(10):
            log("RL Score: " + str(RL_database[i]['RL_score']))

    

def log(message, file_path="db.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)   

def set_database():
    Board_Generator_RL(size = 5500)
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

    
    




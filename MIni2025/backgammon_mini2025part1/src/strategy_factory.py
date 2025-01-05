from src.compare_all_moves_strategy import CompareAllMovesSimple
from src.strategies import MoveFurthestBackStrategy, HumanStrategy, MoveRandomPiece
from src.MinMax import MinMax_shayOren
from src.MinMax_copy import MinMax_shayOren2, MinMax_shayOren3, MinMax_shayOren4, MinMax_shayOren5, MinMax_shayOren6, MinMax_shayOren7, MinMax_shayOren8, MinMax_shayOren9
from src.MCTS import MCTS_shayOren
from src.MCTS2 import MCTS_shayOren2, MCTS_shayOren3, MCTS_shayOren4, MCTS_shayOren5, MCTS_shayOren6, MCTS_shayOren7
class StrategyFactory:
    @staticmethod
    def create_by_name(strategy_name):
        for strategy in StrategyFactory.get_all():
            if strategy.__name__ == strategy_name:
                return strategy()

        raise Exception("Cannot find strategy %s" % strategy_name)

    @staticmethod
    def get_all():
        strategies = [
            MCTS_shayOren,
            MoveRandomPiece,
            MoveFurthestBackStrategy,
            CompareAllMovesSimple,
            HumanStrategy, MCTS_shayOren2, MCTS_shayOren3, MCTS_shayOren4, MCTS_shayOren5, MCTS_shayOren6, MCTS_shayOren7
        ]
        """MinMax_shayOren2,
            MinMax_shayOren3,
            MinMax_shayOren4,
            MinMax_shayOren5,
            MinMax_shayOren6,
            MinMax_shayOren7,
            MinMax_shayOren8,
            MinMax_shayOren9,"""
        return strategies

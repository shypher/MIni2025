from src.compare_all_moves_strategy import CompareAllMovesSimple
from src.strategies import MoveFurthestBackStrategy, HumanStrategy, MoveRandomPiece
from src.MinMax import MinMax_shayOren
from src.MinMax_copy import MinMax_shayOren2, MinMax_shayOren3, MinMax_shayOren4, MinMax_shayOren5, MinMax_shayOren6, MinMax_shayOren7, MinMax_shayOren8, MinMax_shayOren9

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
            MinMax_shayOren,
            MinMax_shayOren2,
            MinMax_shayOren3,
            MinMax_shayOren4,
            MinMax_shayOren5,
            MinMax_shayOren6,
            MinMax_shayOren7,
            MinMax_shayOren8,
            MinMax_shayOren9,
            MoveRandomPiece,
            MoveFurthestBackStrategy,
            CompareAllMovesSimple,
            HumanStrategy,
        ]
        return strategies

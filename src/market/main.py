from data.manager import DataManager
from model import preprocessing
from routines.routine import Routine_next_day, Routine_several_days
from model.model import MLP_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from visualization import visualization
from bot import bot

def main() -> None:
    """
    main function
    """

    # routine1 = Routine_next_day()
    routine2 = Routine_several_days()
    # hermes = bot.bot_telegram()



if __name__ == '__main__':
    main()
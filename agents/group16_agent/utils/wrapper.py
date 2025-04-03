# python objects to store opponent information
import pandas as pd
import logging
from tudelft_utilities_logging.ReportToLogger import ReportToLogger


class Opponent:
    def __init__(self, result=0, finalUtility=0, offerVariance=[], name=""):
        self.result = result
        self.finalUtility = finalUtility
        self.offerVariance = offerVariance
        self.name = name

    def save(self):
        save_opponent_data(self)

    def normalize(self):
        # not the behaviour, will take all the raw data and make it into statistics
        return self.result


def get_opponent_data(him, name):

    him.logger.log(logging.INFO, "\n\n\n\n\n\nfunction called as wanted \n\n\n\n\n\n")
    file_path = f"saved/{name}.plk"

    try:
        opponent = pd.read_pickle(file_path)
        if not isinstance(opponent, Opponent):
            raise ValueError("Deserialized object is not of type Opponent")
        opponent.normalize()
    except (FileNotFoundError, ValueError, Exception):
        print(f"File not found or invalid data. Creating default Opponent for {name}.")
        opponent = Opponent(name=name)
        pd.to_pickle(opponent, file_path)

    return opponent


def save_opponent_data(opponent):
    if isinstance(opponent, Opponent):
        file_path = f"saved/{opponent.name}.plk"
        pd.to_pickle(opponent, file_path)
    else:
        print("Non opponent saved")

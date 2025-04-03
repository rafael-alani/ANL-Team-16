# python objects to store opponent information
import pandas as pd
import os


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


def get_opponent_data(savepath, name):

    file_path = savepath + name+".plk"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    try:
        opponent = pd.read_pickle(file_path)
        if not isinstance(opponent, Opponent):
            raise ValueError("Deserialized object is not of type Opponent")
        opponent.normalize()
    except (FileNotFoundError, ValueError, Exception):
        print(f"File not found or invalid data. Creating default Opponent for {name}. at path: ", savepath)
        opponent = Opponent(name=name)
        pd.to_pickle(opponent, file_path)

    return opponent


def save_opponent_data(savepath, opponent):
    if(opponent == None):
        print("we have a problem opponent is None")
        return
    if isinstance(opponent, Opponent):
        print(f"we are saving {opponent.name}")
        file_path = savepath + opponent.name + ".plk"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        pd.to_pickle(opponent, file_path)
    else:
        print("Non opponent saved")

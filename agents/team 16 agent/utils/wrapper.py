# python objects to store opponent information
import pandas


class Opponent:
    def __init__(self, result = 0, finalUtility = 0, offerVariance = []):
        self.result = result
        self.finalUtility = finalUtility
        self.offerVariance = offerVariance


def get_opponent_data(name):
    file_path = f"saved/{name}.plk"

    try:
        opponent = pd.read_pickle(file_path)
        if not isinstance(opponent, Opponent):
            raise ValueError("Deserialized object is not of type Opponent")
    except (FileNotFoundError, ValueError, Exception):
        print(f"File not found or invalid data. Creating default Opponent for {name}.")
        opponent = Opponent(name)
        pd.to_pickle(opponent, file_path)

    return opponent

# python objects to store opponent information
import pandas as pd
import os


class Opponent:
    def __init__(self, result=0, finalUtility=0, offerVariance=[], name="", sessions=None):
        self.result = result
        self.finalUtility = finalUtility
        self.offerVariance = offerVariance
        self.name = name
        self.sessions = sessions if sessions is not None else []

    def add_session(self, session_data):
        """Add a new session data entry to the sessions list"""
        if self.sessions is None:
            self.sessions = []
        self.sessions.append(session_data)

    def save(self, savepath):
        save_opponent_data(savepath, self)

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


def create_and_save_session_data(opponent, savepath, progress, utility_at_finish, did_accept, opponent_model=None):
    """Create and save session data for an opponent
    
    Args:
        opponent: The Opponent object to update
        savepath: Path to save the opponent data
        progress: Current negotiation progress
        utility_at_finish: Utility of the final agreement (0 if no agreement)
        did_accept: Whether we accepted the opponent's offer
        opponent_model: Optional OpponentModel to get current parameters from
    """
    if opponent is None:
        print("Cannot save session data: opponent is None")
        return
        
    # Get default values
    min_util = 0.7  # Threshold for a good deal
    top_bids_percentage = 1/300
    force_accept_at_remaining_turns = 1
    
    # Get values from opponent model if available
    if opponent_model:
        top_bids_percentage = getattr(opponent_model, 'top_bids_percentage', top_bids_percentage)
        force_accept_at_remaining_turns = getattr(opponent_model, 'force_accept_at_remaining_turns', force_accept_at_remaining_turns)
    
    # Create session data dictionary
    session_data = {
        "progressAtFinish": progress,
        "utilityAtFinish": utility_at_finish,
        "didAccept": did_accept,
        "isGood": utility_at_finish >= min_util,
        "topBidsPercentage": top_bids_percentage,
        "forceAcceptAtRemainingTurns": force_accept_at_remaining_turns
    }
    
    # Add session data to opponent
    opponent.add_session(session_data)
    
    # Save opponent data
    save_opponent_data(savepath, opponent)

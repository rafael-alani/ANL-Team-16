import logging
from random import randint, uniform, choice
from statistics import variance, mean
from time import time
from typing import cast, List

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from numpy import floor
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel
from .utils import wrapper


class Group16Agent(DefaultParty):
    """
    The amazing Python geniusweb agent made by team 16.

    Should store general information so that geniuse web works
    Should store information about the current exchange that isn't related to the opponent
    Opponent model should store information about the opponent
    Opponent/Wrapper should store information about the opponent that we want persistent between encounters

    Key features:
    - Uses variance in opponent bids to estimate concession willingness
    - Calculates target utility using Ut = Umax - (Umax - Umin) * t formula - inspired by Gahboninho
    - Integrates session data from previous negotiations
    - Uses random selection of bids above target utility( may be changed to a more sophisticated approach)
    """

    def __init__(self):
        super().__init__()
        self.find_bid_result = None
        self.best_bid = None
        self.bids_with_utilities = None
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None
        self.got_opponent = False

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.opponent = None

        self.all_bids: AllBidsList = None
        
        # Gahboninho utility parameters
        self.round_count: int = 0
        self.opponent_utilities: List[float] = []
        self.concession_rate: float = 0.0
        self.opponent_utility_variance: float = 0.0
        self.probing_phase_complete: bool = False
        self.max_target_utility: float = 0.95
        self.min_target_utility: float = 0.7
        
        # deadline management
        self.avg_time_per_round = None
        self.round_times = []
        self.last_time = None
        
        # current session tracking
        self.utility_at_finish: float = 0.0
        self.did_accept: bool = False
        
        
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            
            
            self.all_bids = AllBidsList(self.domain)
            
            
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            # RAFA: check if agreement reached
            agreements = cast(Finished, data).getAgreements()
            if len(agreements.getMap()) > 0:
                agreed_bid = agreements.getMap()[self.me]
                # CHANGE: name of var
                self.utility_at_finish = float(self.profile.getUtility(agreed_bid))
                self.logger.log(logging.INFO, f"Agreement reached with utility: {self.utility_at_finish}")
            else:
                self.utility_at_finish = 0.0
                self.logger.log(logging.INFO, "No agreement reached")
            
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Team 16's agent, the best agent in the tournament!"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()
            
            # Get our utility for this bid and send it to the model
            our_utility = float(self.profile.getUtility(bid))

            # update opponent model with bid and our utility
            self.opponent_model.update(bid, our_utility)
            
            # best bid the opponent made
            if self.best_bid is None or our_utility > float(self.profile.getUtility(self.best_bid)):
                self.best_bid = bid
            
            # previous utilities of opponent
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            self.opponent_utilities.append(opponent_utility)
            
            # set bid as last received
            self.last_received_bid = bid
            
            # HARDCODED: 5 bids are enough to calculate the metrics
            # TODO: Look into a better approach
            # still update even after probing is done
            if len(self.opponent_utilities) >= 5 and not self.probing_phase_complete:
                self.update_concession_metrics()
                self.probing_phase_complete = True
            elif len(self.opponent_utilities) >= 5:
                self.update_concession_metrics()

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # keep track of:
        # - time per round
        # - round count
        # - avg time per round recently
        current_time = time()
        if self.last_time is not None:
            round_time = current_time - self.last_time
            self.round_times.append(round_time)
            if len(self.round_times) >= 3:
                self.avg_time_per_round = mean(self.round_times[-3:])
        self.last_time = current_time

        self.round_count += 1
        
        # ONLY try to load opponent data if we know who the opponent is, might be wrong
        if self.other is not None and not self.got_opponent:
            self.opponent = wrapper.get_opponent_data(self.parameters.get("storage_dir"), self.other)
            
            if self.opponent_model is not None and self.opponent.sessions:
                self.opponent_model.learn_from_past_sessions(self.opponent.sessions)
            self.got_opponent = True
            
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
            self.did_accept = True
            self.logger.log(logging.INFO, f"Accepting bid with utility: {float(self.profile.getUtility(self.last_received_bid))}")
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)
            self.logger.log(logging.INFO, f"Offering bid with utility: {float(self.profile.getUtility(bid))}")

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        # problem with  trying to save opponent data if we don't have an opponent response yet
        if self.other is not None and self.opponent is not None:
            wrapper.create_and_save_session_data(
                opponent=self.opponent,
                savepath=self.parameters.get("storage_dir"),
                progress=self.progress.get(time() * 1000),
                utility_at_finish=self.utility_at_finish,
                did_accept=self.did_accept,
                opponent_model=self.opponent_model
            )
        else:
            self.logger.log(logging.INFO, "No opponent data to save (opponent unknown or no model created)")
        
        self.got_opponent = False
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)


    def update_concession_metrics(self):
        """Update metrics about the opponent's concession behavior
        
        We use this measure opponent's willingness to concede
        based on variance in utilities and concession rate similar to Gahboninho
        """
        try:
            self.opponent_utility_variance = variance(self.opponent_utilities)
            
            # simple approach: how much did opponent concede on average between consecutive bids
            total_decrease = 0.0
            decreases_count = 0
            
            for i in range(1, len(self.opponent_utilities)):
                diff = self.opponent_utilities[i-1] - self.opponent_utilities[i]
                if diff > 0:
                    total_decrease += diff
                    decreases_count += 1
            
            self.concession_rate = total_decrease / max(1, decreases_count)
            
            self.logger.log(logging.INFO, f"Opponent utility variance: {self.opponent_utility_variance}")
            self.logger.log(logging.INFO, f"Opponent concession rate: {self.concession_rate}")
            
        except Exception as e:
            self.logger.log(logging.WARNING, f"Error calculating concession metrics: {e}")

    def calculate_target_utility(self, progress: float) -> float:
        """Calculate target utility using Gahboninho's formula:
        Ut = Umax - (Umax - Umin) * t
        for now we only use constant values
        this should be later changed
        Adjusted based on opponent's concession behavior
        """
        # # Base values 
        umax = self.max_target_utility
        umin = self.min_target_utility
        
        # if opponent doesn't concede much (low variance and rate), we should be more willing to concede
        if self.opponent_utility_variance < 0.01 or self.concession_rate < 0.02:
            concession_factor = 1.1
            umin = max(0.65, umin - 0.05)
        elif self.opponent_utility_variance > 0.03 or self.concession_rate > 0.05:

            concession_factor = 0.8
            umin = min(0.85, umin + 0.05)
        else:
            concession_factor = 1.0
        
        # Gahboninho's formula with the modified concession factor
        target = umax - (umax - umin) * progress * concession_factor
        
        self.max_target_utility = umax
        self.min_target_utility = umin

        # Cap the minimum
        return max(target, umin)

    def accept_condition(self, bid: Bid) -> bool:
        """Determine whether to accept opponent's bid
        ???
        Uses DreamTeam-style dynamic thresholds and Gahboninho's target utility
        """
        if bid is None:
            return False

        utility = float(self.profile.getUtility(bid))
        
        # best opponent bid so far
        if self.best_bid is None or float(self.profile.getUtility(self.best_bid)) < utility:
            self.best_bid = bid

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)
        
        target_utility = self.calculate_target_utility(progress)
        
        # dynamic thresholds
        threshold = 0.98
        light_threshold = 0.95
        if self.avg_time_per_round is not None:
            threshold = 1 - 1000 * self.opponent_model.force_accept_at_remaining_turns  * self.avg_time_per_round / self.progress.getDuration()
            light_threshold = 1 - 5000 * self.opponent_model.force_accept_at_remaining_turns_light * self.avg_time_per_round / self.progress.getDuration()
        
        # Accept conditions
        conditions = [
            utility > 0.9,                                   # amazing offer, just accept
            utility >= target_utility,                       # meets target
            progress > threshold,                            # close to deadline
            progress > light_threshold and utility >= target_utility - 0.05  # close to deadline and we have someting with close
            # to our target uility
        ]
        return any(conditions)

    def find_bid(self) -> Bid:
        # NOTE
        # Use the opponent model to improve bidding strategy:
        # 1. self.opponent_model.get_opponent_type() - Returns opponent type (HARDHEADED, CONCEDER, NEUTRAL)
        # 2. self.opponent_model.get_top_issues(3) - Returns top 3 issues important to opponent as [(issue_id, weight),...]
        # 3. self.opponent_model.get_predicted_utility(bid) - Estimate opponent's utility for a bid
        # 4. self.opponent_model.best_bid_for_us - Best bid received (highest utility for us)
        
    
        """
        - Uses probing in early negotiation
        - Uses target utility formula
        - Randomly selects bids above target utility
        - Considers opponent's best bid in late stages
        """

        # current progress of the negotiation (0 to 1 scale)
        progress = self.progress.get(time() * 1000)
        
        # highest bids possible
        if self.round_count < 5:
            # this could also be constant
            target_utility = max(0.9, 0.95 - self.round_count * 0.01)
        else:
            target_utility = self.calculate_target_utility(progress)
        
        # start considering opponent's best bid
        light_threshold = 0.95
        if self.avg_time_per_round is not None:
            light_threshold = 1 - 5000 * self.opponent_model.force_accept_at_remaining_turns_light * self.avg_time_per_round / self.progress.getDuration()
            
        if progress > light_threshold and self.best_bid is not None:
            best_bid_utility = float(self.profile.getUtility(self.best_bid))
            if best_bid_utility >= target_utility - 0.1:
                return self.best_bid
        
        # Calculate bids with utilities if not done yet
        if self.bids_with_utilities is None:
            self.bids_with_utilities = []
            for index in range(self.all_bids.size()):
                bid = self.all_bids.get(index)
                bid_utility = float(self.profile.getUtility(bid))
                self.bids_with_utilities.append((bid, bid_utility))
            
            # python sort by utility (highest first)
            self.bids_with_utilities.sort(key=lambda tup: tup[1], reverse=True)
        
        # find all bids above target utility
        eligible_bids = []
        for bid, utility in self.bids_with_utilities:
            if utility >= target_utility:
                eligible_bids.append((bid, utility))
                # this has to be tested
                # TEST
                if len(eligible_bids) >= 100:
                    break
        
        # randomly select one
        if eligible_bids:
            return choice(eligible_bids)[0]
        
        # If no eligible bids found, fallback to a utility-based approach with expanding range
        top_percentage = max(0.01, min(0.2, self.opponent_model.top_bids_percentage + progress * 0.19))
        expanded_top_bids = max(5, floor(self.all_bids.size() * top_percentage))
        next_bid = randint(0, min(expanded_top_bids, len(self.bids_with_utilities)) - 1)
        self.logger.log(logging.INFO, f"Interesting: {top_percentage}, {expanded_top_bids}, {next_bid}")
        
        return self.bids_with_utilities[next_bid][0]
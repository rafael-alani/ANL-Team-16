import logging
from random import randint, uniform, choice
from time import time
from typing import cast

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

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        if(not self.got_opponent):
            self.opponent = wrapper.get_opponent_data(self.parameters.get("storage_dir"), self.other)
            self.got_opponent = True
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        wrapper.save_opponent_data(self.parameters.get("storage_dir"), self.opponent)
        self.got_opponent = False
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # Keep track of the best bid the opponent made so far
        utility = self.profile.getUtility(bid)
        if self.best_bid is None or self.profile.getUtility(self.best_bid) < utility:
            self.best_bid = bid

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.profile.getUtility(bid) > 0.8,
            progress > 0.95,
        ]
        return all(conditions)

    def find_bid(self) -> Bid:
        """
        Determines the next bid to offer.
        - If the opponent's best bid's utility is sufficient, offer it
        - Else, start by offering bids from the top 1% ranked by utility, expand the bid range as time progresses
        """

        # Get the current progress of the negotiation (0 to 1 scale)
        progress = self.progress.get(time() * 1000)

        # Calculate the minimum utility threshold dynamically based on progress
        # If the opponent's best bid meets the dynamically decreasing utility requirement, offer it
        # Add randomness and variation to the threshold to make us less predictable
        if self.best_bid is not None:
            #min_utility_threshold = max(0.5, 1.4 - 0.9 * progress)
            random_variation = uniform(-0.02, 0.02)
            random_strategy = choice(['linear', 'quadratic'])
            if random_strategy == 'linear':
                min_utility_threshold = max(0.6, min(1.0, -0.4 * progress + 1 + random_variation))
            else:
                min_utility_threshold = max(0.6, min(1.0, -0.4 * (progress ** 2) + 1 + random_variation))

            best_bid_utility = float(self.profile.getUtility(self.best_bid))

            if best_bid_utility >= min_utility_threshold:
                return self.best_bid

        # Retrieve all possible bids in the domain
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)
        num_of_bids = all_bids.size()

        # If bids with utilities haven't been calculated yet, compute them
        if self.bids_with_utilities is None:
            self.bids_with_utilities = []

            # Calculate utility for each bid and store them in a list
            for index in range(num_of_bids):
                bid = all_bids.get(index)
                bid_utility = float(self.profile.getUtility(bid))
                self.bids_with_utilities.append((bid, bid_utility))

            # Sort bids by utility from high to low
            self.bids_with_utilities.sort(key=lambda tup: tup[1], reverse=True)

        # Expand the range of acceptable bids over time (starts at 1% and increases gradually up to 20%)
        increasing_percentage = min(0.01 + progress * 0.19, 0.2)
        expanded_top_bids = max(5, floor(num_of_bids * increasing_percentage))

        # Dynamically decrease threshold: as time progresses, the threshold lowers, making concessions more likely
        #dynamic_threshold = max(0.5, 1 - progress * 0.5)

        # If progress exceeds the threshold, offer the best bid from the opponent
        #if progress > dynamic_threshold and self.best_bid is not None:
        #    return self.best_bid

        # Randomly select a bid from the expanded top bids range
        next_bid = randint(0, expanded_top_bids - 1)
        self.find_bid_result = self.bids_with_utilities[next_bid][0]
        return self.bids_with_utilities[next_bid][0]
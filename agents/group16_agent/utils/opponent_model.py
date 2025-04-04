from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value


class OpponentModel:
    def __init__(self, domain: Domain):
        self.offers = []
        self.domain = domain
        
        # Track best bid for us (highest utility for our agent)
        self.best_bid_for_us = None
        self.best_bid_utility = 0.0
        
        # Issue estimators to track opponent preferences
        self.issue_estimators = {
            i: IssueEstimator(v) for i, v in domain.getIssuesValues().items()
        }
        
        # Track opponent strategy type
        self.bid_count = 0
        self.concession_rate = 0.0
        self.repeated_bids = defaultdict(int)

        # learn from previous mistakes
        self.force_accept_at_remaining_turns = 1
        self.force_accept_at_remaining_turns_light = 1
        
        self.hard_accept_at_turn_X = 1
        self.soft_accept_at_turn_X = 1
        self.top_bids_percentage = 1/300
        
        # Track history of opponent utilities to analyze concession
        self.opponent_utilities = []

    def update(self, bid: Bid, our_utility: float = None):
        """Update the opponent model with a new bid
        
        Args:
            bid (Bid): New bid from opponent
            our_utility (float, optional): Our utility for this bid
        """
        # Track all received bids
        self.offers.append(bid)
        self.bid_count += 1
        
        # Count repeated bids (use string representation for hashability)
        bid_str = str(bid)
        self.repeated_bids[bid_str] += 1
        
        # Update best bid for us if applicable
        if our_utility is not None and (self.best_bid_for_us is None or our_utility > self.best_bid_utility):
            self.best_bid_for_us = bid
            self.best_bid_utility = our_utility
        
        # Update all issue estimators
        for issue_id, issue_estimator in self.issue_estimators.items():
            issue_estimator.update(bid.getValue(issue_id))
        
        # Calculate opponent utility and track it
        opponent_utility = self.get_predicted_utility(bid)
        self.opponent_utilities.append(opponent_utility)
        
        # Update concession rate if we have at least 2 bids
        if len(self.opponent_utilities) >= 2:
            # Simple concession rate - average decrease in utility between consecutive bids
            decreases = []
            for i in range(1, len(self.opponent_utilities)):
                diff = self.opponent_utilities[i-1] - self.opponent_utilities[i]
                if diff > 0:  # Only count decreases in utility (actual concessions)
                    decreases.append(diff)
            
            if decreases:
                self.concession_rate = sum(decreases) / len(decreases)

    def get_predicted_utility(self, bid: Bid) -> float:
        """Predict the opponent's utility for a bid
        
        Args:
            bid (Bid): Bid to evaluate
            
        Returns:
            float: Predicted utility between 0 and 1
        """
        if len(self.offers) == 0 or bid is None:
            return 0

        # initiate
        total_issue_weight = 0.0
        value_utilities = []
        issue_weights = []

        for issue_id, issue_estimator in self.issue_estimators.items():
            # get the value that is set for this issue in the bid
            value: Value = bid.getValue(issue_id)

            # collect both the predicted weight for the issue and
            # predicted utility of the value within this issue
            value_utilities.append(issue_estimator.get_value_utility(value))
            issue_weights.append(issue_estimator.weight)

            total_issue_weight += issue_estimator.weight

        # normalise the issue weights such that the sum is 1.0
        if total_issue_weight == 0.0:
            issue_weights = [1 / len(issue_weights) for _ in issue_weights]
        else:
            issue_weights = [iw / total_issue_weight for iw in issue_weights]

        # calculate predicted utility by multiplying all value utilities with their issue weight
        predicted_utility = sum(
            [iw * vu for iw, vu in zip(issue_weights, value_utilities)]
        )

        return predicted_utility
    
    def get_opponent_type(self) -> str:
        """Identify opponent negotiation strategy type
        
        Returns:
            str: Strategy type (HARDHEADED, CONCEDER, or UNKNOWN)
        """
        if self.bid_count < 3:
            return "UNKNOWN"
        
        # Check for hardheaded opponent (low concession rate, many repeated bids)
        unique_bids = len(self.repeated_bids)
        bid_repetition_ratio = unique_bids / self.bid_count
        
        if self.concession_rate < 0.02 or bid_repetition_ratio < 0.5:
            return "HARDHEADED"
        elif self.concession_rate > 0.05:
            return "CONCEDER"
        else:
            return "NEUTRAL"
    
    def get_concession_rate(self) -> float:
        """Get the opponent's concession rate
        
        Returns:
            float: Concession rate (higher means more concessions)
        """
        return self.concession_rate
    
    def get_top_issues(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get the top n most important issues for the opponent
        
        Args:
            n (int, optional): Number of issues to return. Defaults to 3.
            
        Returns:
            List[Tuple[str, float]]: List of (issue_id, weight) tuples
        """
        # Get normalized issue weights
        total_weight = sum(est.weight for est in self.issue_estimators.values())
        
        if total_weight == 0:
            # Equal weights if no data yet
            equal_weight = 1.0 / len(self.issue_estimators)
            issues = [(issue_id, equal_weight) for issue_id in self.issue_estimators.keys()]
        else:
            # Normalize weights
            issues = [(issue_id, est.weight / total_weight) 
                    for issue_id, est in self.issue_estimators.items()]
        
        # Sort by weight and return top n
        issues.sort(key=lambda x: x[1], reverse=True)
        return issues[:n]
    
    # TODO: Add methods to save/load opponent data for persistent learning

    def learn_from_past_sessions(self, sessions: list):
        hard_accept_levels = [0, 0, 1, 1.1]
        soft_accept_levels = [0, 1, 1.1]
        top_bids_levels = [1 / 300, 1 / 100, 1 / 30]
        
        # fully failed
        failed_sessions_count = 0
        for session in sessions:
            # Check if session failed (utility == 0)
            if isinstance(session, dict) and session.get("utilityAtFinish", 1) == 0:
                failed_sessions_count += 1
        
        # low utility
        low_utility_sessions_count = 0
        for session in sessions:
            # Check if session had low utility (utility < 0.5)
            if isinstance(session, dict) and session.get("utilityAtFinish", 1) < 0.5:
                low_utility_sessions_count += 1
        
        # hard accept based on previous failed sessions
        if failed_sessions_count >= len(hard_accept_levels):
            accept_index = len(hard_accept_levels) - 1
        else:
            accept_index = failed_sessions_count
        #self.hard_accept_at_turn_X = hard_accept_levels[accept_index]
        self.force_accept_at_remaining_turns = hard_accept_levels[accept_index]
        
        # soft 
        if failed_sessions_count >= len(soft_accept_levels):
            light_accept_index = len(soft_accept_levels) - 1
        else:
            light_accept_index = failed_sessions_count
        #self.soft_accept_at_turn_X = soft_accept_levels[light_accept_index]
        self.force_accept_at_remaining_turns_light = soft_accept_levels[light_accept_index]
        
        # Set top_bids_percentage based on low utility sessions
        if low_utility_sessions_count >= len(top_bids_levels):
            top_bids_index = len(top_bids_levels) - 1
        else:
            top_bids_index = low_utility_sessions_count
        self.top_bids_percentage = top_bids_levels[top_bids_index]

class IssueEstimator:
    def __init__(self, value_set: DiscreteValueSet):
        if not isinstance(value_set, DiscreteValueSet):
            raise TypeError(
                "This issue estimator only supports issues with discrete values"
            )

        self.bids_received = 0
        self.max_value_count = 0
        self.num_values = value_set.size()
        self.value_trackers = defaultdict(ValueEstimator)
        self.weight = 0

    def update(self, value: Value):
        self.bids_received += 1

        # get the value tracker of the value that is offered
        value_tracker = self.value_trackers[value]

        # register that this value was offered
        value_tracker.update()

        # update the count of the most common offered value
        self.max_value_count = max([value_tracker.count, self.max_value_count])

        # update predicted issue weight
        # the intuition here is that if the values of the receiverd offers spread out over all
        # possible values, then this issue is likely not important to the opponent (weight == 0.0).
        # If all received offers proposed the same value for this issue,
        # then the predicted issue weight == 1.0
        equal_shares = self.bids_received / self.num_values
        self.weight = (self.max_value_count - equal_shares) / (
            self.bids_received - equal_shares
        )

        # recalculate all value utilities
        for value_tracker in self.value_trackers.values():
            value_tracker.recalculate_utility(self.max_value_count, self.weight)

    def get_value_utility(self, value: Value):
        if value in self.value_trackers:
            return self.value_trackers[value].utility

        return 0


class ValueEstimator:
    def __init__(self):
        self.count = 0
        self.utility = 0

    def update(self):
        self.count += 1

    def recalculate_utility(self, max_value_count: int, weight: float):
        if weight < 1:
            mod_value_count = ((self.count + 1) ** (1 - weight)) - 1
            mod_max_value_count = ((max_value_count + 1) ** (1 - weight)) - 1

            self.utility = mod_value_count / mod_max_value_count
        else:
            self.utility = 1

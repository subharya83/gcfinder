import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import math
import random

@dataclass
class Validator:
    """Represents a validator in the blockchain network."""
    validator_id: str
    stake: float
    reputation: float
    locked_stake: float = 0.0
    last_validation: Optional[datetime] = None

@dataclass
class Contribution:
    """Represents a contribution submitted for validation."""
    contribution_id: str
    user_id: str
    data: Dict
    stake_amount: float
    timestamp: datetime
    ipfs_hash: Optional[str] = None
    validation_status: str = "pending"  # pending, validated, rejected
    votes: List[Dict] = None
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = []

@dataclass
class ValidationVote:
    """Represents a validator's vote on a contribution."""
    validator_id: str
    contribution_id: str
    vote: bool  # True for valid, False for invalid
    weight: float
    timestamp: datetime

class BlockchainTrustMechanism:
    def __init__(self, reputation_learning_rate: float = 0.1, accuracy_weight: float = 0.6):
        """
        Initialize the blockchain trust mechanism.
        
        Args:
            reputation_learning_rate: Alpha parameter for reputation updates
            accuracy_weight: Beta parameter weighting accuracy vs consensus
        """
        self.validators: Dict[str, Validator] = {}
        self.contributions: Dict[str, Contribution] = {}
        self.reputation_learning_rate = reputation_learning_rate
        self.accuracy_weight = accuracy_weight
        self.consensus_threshold = 2/3  # Two-thirds majority requirement
        
    def register_validator(self, validator_id: str, initial_stake: float, initial_reputation: float = 0.5):
        """Register a new validator in the system."""
        validator = Validator(
            validator_id=validator_id,
            stake=initial_stake,
            reputation=initial_reputation
        )
        self.validators[validator_id] = validator
        return validator
    
    def calculate_validation_weight(self, validator_id: str, rho: float = 0.5) -> float:
        """
        Calculate validator weight using:
        Validation_weight(v) = Stake(v) · Reputation(v)^ρ
        """
        if validator_id not in self.validators:
            return 0.0
        
        validator = self.validators[validator_id]
        weight = validator.stake * (validator.reputation ** rho)
        return weight
    
    def calculate_voting_weight(self, validator_id: str) -> float:
        """
        Calculate voting weight with diminishing returns:
        weight = pow(stake, 0.5) * pow(reputation, 0.3)
        """
        if validator_id not in self.validators:
            return 0.0
        
        validator = self.validators[validator_id]
        weight = (validator.stake ** 0.5) * (validator.reputation ** 0.3)
        return weight
    
    def generate_contribution_id(self, user_id: str, data: Dict) -> str:
        """Generate unique contribution identifier using cryptographic hashing."""
        timestamp = str(time.time())
        content = json.dumps(data, sort_keys=True)
        hash_input = f"{user_id}_{timestamp}_{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def simulate_ipfs_storage(self, data: Dict) -> str:
        """Simulate IPFS storage and return hash."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def submit_contribution(self, user_id: str, data: Dict, stake_amount: float) -> str:
        """
        Submit a contribution for validation.
        Locks the stake amount during validation process.
        """
        contribution_id = self.generate_contribution_id(user_id, data)
        ipfs_hash = self.simulate_ipfs_storage(data)
        
        contribution = Contribution(
            contribution_id=contribution_id,
            user_id=user_id,
            data=data,
            stake_amount=stake_amount,
            timestamp=datetime.now(),
            ipfs_hash=ipfs_hash
        )
        
        self.contributions[contribution_id] = contribution
        return contribution_id
    
    def submit_validation_vote(self, validator_id: str, contribution_id: str, vote: bool) -> bool:
        """
        Submit a validation vote from a validator.
        Returns True if vote was successfully recorded.
        """
        if validator_id not in self.validators:
            return False
        
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        if contribution.validation_status != "pending":
            return False  # Already validated
        
        # Calculate voting weight
        weight = self.calculate_voting_weight(validator_id)
        
        # Create validation vote
        validation_vote = ValidationVote(
            validator_id=validator_id,
            contribution_id=contribution_id,
            vote=vote,
            weight=weight,
            timestamp=datetime.now()
        )
        
        # Add vote to contribution
        contribution.votes.append(asdict(validation_vote))
        
        # Update validator's last validation time
        self.validators[validator_id].last_validation = datetime.now()
        
        # Check if consensus is reached
        self._check_consensus(contribution_id)
        
        return True
    
    def _check_consensus(self, contribution_id: str) -> bool:
        """
        Check if consensus is reached (two-thirds majority by weight).
        """
        contribution = self.contributions[contribution_id]
        
        if not contribution.votes:
            return False
        
        # Calculate total weight and positive weight
        total_weight = sum(vote['weight'] for vote in contribution.votes)
        positive_weight = sum(vote['weight'] for vote in contribution.votes if vote['vote'])
        
        if total_weight == 0:
            return False
        
        # Check if consensus threshold is met
        positive_ratio = positive_weight / total_weight
        
        if positive_ratio >= self.consensus_threshold:
            # Consensus for validation
            contribution.validation_status = "validated"
            self._finalize_validation(contribution_id, True)
            return True
        elif (1 - positive_ratio) >= self.consensus_threshold:
            # Consensus for rejection
            contribution.validation_status = "rejected"
            self._finalize_validation(contribution_id, False)
            return True
        
        return False  # No consensus yet
    
    def _finalize_validation(self, contribution_id: str, is_validated: bool):
        """
        Finalize validation results and update reputations.
        """
        contribution = self.contributions[contribution_id]
        
        # Update contributor reputation
        if is_validated:
            # Successful contribution: +0.05 reputation
            self._update_user_reputation(contribution.user_id, 0.05)
        else:
            # Rejected contribution: -0.1 reputation
            self._update_user_reputation(contribution.user_id, -0.10)
        
        # Update validator reputations based on consensus alignment
        consensus_vote = is_validated
        
        for vote_data in contribution.votes:
            validator_id = vote_data['validator_id']
            vote = vote_data['vote']
            
            if vote == consensus_vote:
                # Validator aligned with consensus: +0.02 reputation
                self._update_validator_reputation(validator_id, 0.02)
            else:
                # Validator voted against consensus: -0.05 reputation
                self._update_validator_reputation(validator_id, -0.05)
        
        # Handle reward distribution
        self._distribute_rewards(contribution_id, is_validated)
    
    def _update_user_reputation(self, user_id: str, reputation_change: float):
        """Update user reputation (simplified - would integrate with user management system)."""
        # In a full implementation, this would update the user's reputation
        # in the crowdsourcing system
        pass
    
    def _update_validator_reputation(self, validator_id: str, reputation_change: float):
        """
        Update validator reputation using the formula:
        Reputation_new = (1-α)·Reputation_old + α·(β·Accuracy + (1-β)·Consensus_agreement)
        """
        if validator_id not in self.validators:
            return
        
        validator = self.validators[validator_id]
        old_reputation = validator.reputation
        
        # For this simplified update, we'll just add the change
        # In a full implementation, this would use the complete formula
        new_reputation = old_reputation + (self.reputation_learning_rate * reputation_change)
        
        # Clamp reputation to [0, 1]
        validator.reputation = max(0.0, min(1.0, new_reputation))
    
    def calculate_base_reward(self, contribution: Contribution) -> float:
        """Calculate base reward for a contribution."""
        # Base reward could depend on contribution type, complexity, etc.
        return 10.0  # Simplified base reward
    
    def calculate_quality_multiplier(self, user_id: str, validated_contributions: int, accuracy_rate: float) -> float:
        """
        Calculate quality multiplier using:
        Quality_multiplier(u) = 1 + log(1 + Validated_contributions(u)) · Accuracy_rate(u)
        """
        if validated_contributions == 0:
            return 1.0
        
        multiplier = 1 + math.log(1 + validated_contributions) * accuracy_rate
        return multiplier
    
    def calculate_reward(self, contribution_id: str, validated_contributions: int, 
                        accuracy_rate: float, timeliness_factor: float = 1.0) -> float:
        """
        Calculate reward using:
        Reward(u) = Base_reward · Quality_multiplier(u) · Stake_bonus(u) · Timeliness_factor(u)
        """
        contribution = self.contributions[contribution_id]
        
        base_reward = self.calculate_base_reward(contribution)
        quality_multiplier = self.calculate_quality_multiplier(
            contribution.user_id, validated_contributions, accuracy_rate
        )
        
        # Stake bonus (simplified - could be more complex)
        stake_bonus = 1 + (contribution.stake_amount / 100.0)  # Bonus based on stake
        
        total_reward = base_reward * quality_multiplier * stake_bonus * timeliness_factor
        return total_reward
    
    def _distribute_rewards(self, contribution_id: str, is_validated: bool):
        """
        Distribute rewards to contributors and validators based on validation outcome.
        """
        contribution = self.contributions[contribution_id]
        
        if is_validated:
            # Calculate and distribute rewards to successful contributor
            # This would integrate with a token/payment system
            
            # Distribute rewards to validators who voted correctly
            for vote_data in contribution.votes:
                if vote_data['vote'] == True:  # Correct vote for validation
                    validator_reward = 1.0 * vote_data['weight']  # Reward proportional to weight
                    # Distribute validator_reward tokens
        else:
            # Redistribute stake penalties to validators who correctly identified invalid data
            penalty_amount = contribution.stake_amount * 0.1  # 10% penalty
            correct_validators = [v for v in contribution.votes if not v['vote']]
            
            if correct_validators:
                reward_per_validator = penalty_amount / len(correct_validators)
                # Distribute rewards to validators who voted against invalid contribution
    
    def apply_reputation_decay(self, decay_rate: float = 0.01, days_inactive: int = 30):
        """
        Apply reputation decay for inactive users:
        Reputation(t) = Reputation(0) · e^(-δ·inactive_time)
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=days_inactive)
        
        for validator in self.validators.values():
            if validator.last_validation is None or validator.last_validation < cutoff_time:
                # Calculate inactive time in days
                if validator.last_validation:
                    inactive_days = (current_time - validator.last_validation).days
                else:
                    inactive_days = days_inactive
                
                # Apply exponential decay
                decay_factor = math.exp(-decay_rate * inactive_days)
                validator.reputation *= decay_factor
                
                # Ensure reputation doesn't go below a minimum threshold
                validator.reputation = max(0.1, validator.reputation)
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics."""
        total_contributions = len(self.contributions)
        validated_contributions = len([c for c in self.contributions.values() if c.validation_status == "validated"])
        rejected_contributions = len([c for c in self.contributions.values() if c.validation_status == "rejected"])
        pending_contributions = total_contributions - validated_contributions - rejected_contributions
        
        # Validator statistics
        active_validators = len([v for v in self.validators.values() if v.last_validation and 
                               (datetime.now() - v.last_validation).days <= 7])
        
        avg_validator_reputation = sum(v.reputation for v in self.validators.values()) / len(self.validators) if self.validators else 0
        total_stake = sum(v.stake for v in self.validators.values())
        
        return {
            "contributions": {
                "total": total_contributions,
                "validated": validated_contributions,
                "rejected": rejected_contributions,
                "pending": pending_contributions
            },
            "validators": {
                "total": len(self.validators),
                "active_7d": active_validators,
                "average_reputation": avg_validator_reputation,
                "total_stake": total_stake
            },
            "system_metrics": {
                "consensus_threshold": self.consensus_threshold,
                "reputation_learning_rate": self.reputation_learning_rate,
                "accuracy_weight": self.accuracy_weight
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize blockchain trust system
    trust_system = BlockchainTrustMechanism()
    
    # Register validators
    trust_system.register_validator("validator1", initial_stake=100.0, initial_reputation=0.8)
    trust_system.register_validator("validator2", initial_stake=150.0, initial_reputation=0.7)
    trust_system.register_validator("validator3", initial_stake=80.0, initial_reputation=0.9)
    
    # Submit a contribution
    sample_data = {
        "restaurant": "Pizza Palace",
        "menu_item": "Margherita Pizza",
        "price": 15.99,
        "availability": True,
        "image_url": "https://example.com/pizza.jpg"
    }
    
    contribution_id = trust_system.submit_contribution("user1", sample_data, stake_amount=10.0)
    print(f"Submitted contribution: {contribution_id}")
    
    # Validators vote on the contribution
    trust_system.submit_validation_vote("validator1", contribution_id, True)  # Valid
    trust_system.submit_validation_vote("validator2", contribution_id, True)  # Valid
    trust_system.submit_validation_vote("validator3", contribution_id, False)  # Invalid
    
    # Check final validation status
    contribution = trust_system.contributions[contribution_id]
    print(f"Final validation status: {contribution.validation_status}")
    
    # Print system statistics
    stats = trust_system.get_system_statistics()
    print("\nSystem Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Print validator reputations after validation
    print("\nValidator Reputations:")
    for validator_id, validator in trust_system.validators.items():
        print(f"{validator_id}: {validator.reputation:.3f}")
    
    # Apply reputation decay simulation
    print("\nApplying reputation decay...")
    trust_system.apply_reputation_decay()
    
    print("Validator Reputations after decay:")
    for validator_id, validator in trust_system.validators.items():
        print(f"{validator_id}: {validator.reputation:.3f}")

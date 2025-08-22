import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Contribution:
    """Represents a user contribution to the system."""
    user_id: str
    content: Dict
    timestamp: datetime
    accuracy_score: Optional[float] = None
    completeness_ratio: float = 0.0
    validation_status: str = "pending"  # pending, validated, rejected

@dataclass
class User:
    """Represents a user in the crowdsourcing system."""
    user_id: str
    reputation: float = 0.5  # Start with neutral reputation
    contributions: List[Contribution] = None
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.contributions is None:
            self.contributions = []
        if self.last_activity is None:
            self.last_activity = datetime.now()

class BiDirectionalCrowdsourcingModel:
    def __init__(self, reputation_decay_factor: float = 0.8, temporal_lambda: float = 0.1):
        """
        Initialize the bi-directional crowdsourcing model.
        
        Args:
            reputation_decay_factor: Alpha value for reputation updates (0-1)
            temporal_lambda: Lambda for temporal relevance calculation
        """
        self.users: Dict[str, User] = {}
        self.contributions: List[Contribution] = []
        self.reputation_decay_factor = reputation_decay_factor
        self.temporal_lambda = temporal_lambda
        
    def calculate_contribution_score(self, contribution: Contribution) -> float:
        """
        Calculate contribution score using the formula:
        Contribution_score(u) = Σ(i=1 to n) (accuracy_i · timeliness_i · completeness_i) / n
        """
        if contribution.accuracy_score is None:
            return 0.0
        
        # Calculate timeliness factor: e^(-λ·age_i)
        current_time = datetime.now()
        age_hours = (current_time - contribution.timestamp).total_seconds() / 3600
        timeliness = math.exp(-self.temporal_lambda * age_hours)
        
        # Completeness ratio is already provided in the contribution
        completeness = contribution.completeness_ratio
        
        # Calculate overall contribution score
        contribution_score = contribution.accuracy_score * timeliness * completeness
        return contribution_score
    
    def update_user_reputation(self, user_id: str, new_contribution_score: float) -> float:
        """
        Update user reputation using the formula:
        Reputation(u,t+1) = α·Reputation(u,t) + (1-α)·Recent_contribution_score(u)
        """
        if user_id not in self.users:
            self.users[user_id] = User(user_id)
        
        user = self.users[user_id]
        old_reputation = user.reputation
        
        # Update reputation
        new_reputation = (self.reputation_decay_factor * old_reputation + 
                         (1 - self.reputation_decay_factor) * new_contribution_score)
        
        # Clamp reputation to [0, 1]
        new_reputation = max(0.0, min(1.0, new_reputation))
        
        user.reputation = new_reputation
        user.last_activity = datetime.now()
        
        return new_reputation
    
    def calculate_completeness_ratio(self, contribution_data: Dict, required_fields: List[str]) -> float:
        """
        Calculate completeness ratio as the proportion of provided fields to required fields.
        """
        provided_fields = sum(1 for field in required_fields if field in contribution_data and contribution_data[field] is not None)
        return provided_fields / len(required_fields) if required_fields else 0.0
    
    def add_contribution(self, user_id: str, content: Dict, required_fields: List[str]) -> str:
        """
        Add a new contribution to the system.
        """
        # Calculate completeness ratio
        completeness_ratio = self.calculate_completeness_ratio(content, required_fields)
        
        # Create contribution
        contribution = Contribution(
            user_id=user_id,
            content=content,
            timestamp=datetime.now(),
            completeness_ratio=completeness_ratio
        )
        
        # Add to contributions list
        self.contributions.append(contribution)
        
        # Add to user's contributions
        if user_id not in self.users:
            self.users[user_id] = User(user_id)
        
        self.users[user_id].contributions.append(contribution)
        
        return f"contribution_{len(self.contributions)}"
    
    def validate_contribution(self, contribution_id: int, accuracy_score: float, is_valid: bool = True):
        """
        Validate a contribution and update user reputation accordingly.
        """
        if 0 <= contribution_id < len(self.contributions):
            contribution = self.contributions[contribution_id]
            contribution.accuracy_score = accuracy_score
            contribution.validation_status = "validated" if is_valid else "rejected"
            
            # Calculate contribution score
            contrib_score = self.calculate_contribution_score(contribution)
            
            # Update user reputation
            self.update_user_reputation(contribution.user_id, contrib_score)
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """
        Get comprehensive statistics for a user.
        """
        if user_id not in self.users:
            return {"error": "User not found"}
        
        user = self.users[user_id]
        
        # Calculate various metrics
        total_contributions = len(user.contributions)
        validated_contributions = len([c for c in user.contributions if c.validation_status == "validated"])
        rejected_contributions = len([c for c in user.contributions if c.validation_status == "rejected"])
        
        # Calculate average accuracy for validated contributions
        validated_contribs = [c for c in user.contributions if c.accuracy_score is not None]
        avg_accuracy = np.mean([c.accuracy_score for c in validated_contribs]) if validated_contribs else 0.0
        
        # Calculate recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_contributions = len([c for c in user.contributions if c.timestamp >= week_ago])
        
        return {
            "user_id": user_id,
            "reputation": user.reputation,
            "total_contributions": total_contributions,
            "validated_contributions": validated_contributions,
            "rejected_contributions": rejected_contributions,
            "pending_contributions": total_contributions - validated_contributions - rejected_contributions,
            "average_accuracy": avg_accuracy,
            "recent_contributions_7d": recent_contributions,
            "last_activity": user.last_activity
        }

class DemandForecastingSystem:
    def __init__(self):
        """Initialize the demand forecasting system."""
        self.historical_data = []
        self.weather_data = []
        self.events_data = []
        self.social_trends_data = []
    
    def predict_demand(self, time_point: datetime, beta_coefficients: List[float] = None) -> float:
        """
        Predict demand using the formula:
        Demand(t) = β₀ + β₁·Historical_avg(t) + β₂·Weather(t) + β₃·Events(t) + β₄·Social_trends(t)
        """
        if beta_coefficients is None:
            beta_coefficients = [10.0, 0.7, 0.2, 0.3, 0.1]  # Default coefficients
        
        beta0, beta1, beta2, beta3, beta4 = beta_coefficients
        
        # Get features for the time point
        historical_avg = self._get_historical_average(time_point)
        weather_factor = self._get_weather_factor(time_point)
        events_factor = self._get_events_factor(time_point)
        social_trends_factor = self._get_social_trends_factor(time_point)
        
        # Calculate demand
        demand = (beta0 + 
                 beta1 * historical_avg + 
                 beta2 * weather_factor + 
                 beta3 * events_factor + 
                 beta4 * social_trends_factor)
        
        return max(0, demand)  # Ensure non-negative demand
    
    def calculate_optimal_stock(self, item_id: str, time_point: datetime, 
                               demand_variance: float = 5.0, service_level: float = 0.95) -> float:
        """
        Calculate optimal stock using:
        Optimal_stock(i,t) = Expected_demand(i,t) + Safety_stock(i) · √(Demand_variance(i))
        """
        expected_demand = self.predict_demand(time_point)
        
        # Calculate safety stock multiplier based on service level
        # Using normal distribution inverse for service level
        if service_level >= 0.99:
            z_score = 2.33
        elif service_level >= 0.95:
            z_score = 1.65
        elif service_level >= 0.90:
            z_score = 1.28
        else:
            z_score = 1.0
        
        safety_stock = z_score * math.sqrt(demand_variance)
        optimal_stock = expected_demand + safety_stock
        
        return optimal_stock
    
    def _get_historical_average(self, time_point: datetime) -> float:
        """Get historical average demand for similar time periods."""
        # Simulate historical data lookup
        hour = time_point.hour
        day_of_week = time_point.weekday()
        
        # Simple pattern: higher demand during lunch (11-14) and dinner (17-21)
        base_demand = 20
        if 11 <= hour <= 14:
            base_demand *= 1.5  # Lunch rush
        elif 17 <= hour <= 21:
            base_demand *= 1.8  # Dinner rush
        elif hour < 7 or hour > 22:
            base_demand *= 0.3  # Low demand hours
        
        # Weekend adjustment
        if day_of_week >= 5:  # Weekend
            base_demand *= 1.2
        
        return base_demand
    
    def _get_weather_factor(self, time_point: datetime) -> float:
        """Get weather impact factor (simplified)."""
        # Simulate weather data - in practice, this would come from weather APIs
        # Return a factor between 0.5 (bad weather) and 1.5 (great weather)
        return 1.0  # Neutral weather
    
    def _get_events_factor(self, time_point: datetime) -> float:
        """Get local events impact factor."""
        # Simulate events data - in practice, this would come from events APIs
        # Check if it's a special day/event
        day_of_week = time_point.weekday()
        hour = time_point.hour
        
        # Higher factor for Friday/Saturday nights
        if day_of_week >= 4 and 18 <= hour <= 23:
            return 1.5
        
        return 1.0  # Normal day
    
    def _get_social_trends_factor(self, time_point: datetime) -> float:
        """Get social media trends impact factor."""
        # Simulate social trends analysis
        # In practice, this would analyze social media mentions, hashtags, etc.
        return 1.0  # Neutral trends

class SystemStabilityMonitor:
    def __init__(self):
        """Initialize system stability monitoring."""
        self.consumer_updates_history = []
        self.business_updates_history = []
    
    def calculate_system_stability(self, consumer_updates: int, business_updates: int, 
                                 total_updates: int) -> float:
        """
        Calculate system stability using:
        System_stability = |Consumer_updates(t) - Business_updates(t)| / Total_updates(t)
        """
        if total_updates == 0:
            return 1.0  # Perfect stability when no updates
        
        stability = abs(consumer_updates - business_updates) / total_updates
        return 1.0 - stability  # Convert to stability score (higher = more stable)
    
    def monitor_convergence(self, actual_states: List[float], reported_states: List[float]) -> float:
        """
        Monitor convergence as the limit of absolute difference between actual and reported states.
        """
        if len(actual_states) != len(reported_states):
            raise ValueError("Actual and reported states must have the same length")
        
        differences = [abs(actual - reported) for actual, reported in zip(actual_states, reported_states)]
        average_difference = np.mean(differences)
        
        # Convert to convergence score (lower difference = higher convergence)
        convergence_score = 1.0 / (1.0 + average_difference)
        return convergence_score

# Example usage
if __name__ == "__main__":
    # Initialize the crowdsourcing system
    crowdsourcing = BiDirectionalCrowdsourcingModel()
    
    # Add some sample contributions
    required_fields = ['restaurant_name', 'menu_item', 'price', 'availability']
    
    # User 1 contribution (high quality)
    contribution1 = {
        'restaurant_name': 'Pizza Palace',
        'menu_item': 'Margherita Pizza',
        'price': '$15.99',
        'availability': 'available'
    }
    
    contrib_id1 = crowdsourcing.add_contribution('user1', contribution1, required_fields)
    crowdsourcing.validate_contribution(0, accuracy_score=0.9, is_valid=True)
    
    # User 2 contribution (partial data)
    contribution2 = {
        'restaurant_name': 'Burger Joint',
        'menu_item': 'Classic Burger',
        # price missing
        'availability': 'available'
    }
    
    contrib_id2 = crowdsourcing.add_contribution('user2', contribution2, required_fields)
    crowdsourcing.validate_contribution(1, accuracy_score=0.6, is_valid=True)
    
    # Print user statistics
    print("User 1 Stats:", crowdsourcing.get_user_statistics('user1'))
    print("User 2 Stats:", crowdsourcing.get_user_statistics('user2'))
    
    # Demonstrate demand forecasting
    forecasting = DemandForecastingSystem()
    future_time = datetime.now() + timedelta(hours=2)
    
    predicted_demand = forecasting.predict_demand(future_time)
    optimal_stock = forecasting.calculate_optimal_stock('pizza_margherita', future_time)
    
    print(f"\nPredicted demand: {predicted_demand:.2f}")
    print(f"Optimal stock level: {optimal_stock:.2f}")
    
    # System stability monitoring
    monitor = SystemStabilityMonitor()
    stability = monitor.calculate_system_stability(consumer_updates=15, business_updates=12, total_updates=27)
    print(f"System stability score: {stability:.3f}")

import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import threading
import hashlib
from functools import lru_cache

@dataclass
class DataSource:
    """Represents an external data source."""
    source_id: str
    name: str
    reliability_score: float
    last_updated: datetime
    api_endpoint: Optional[str] = None
    rate_limit: int = 100  # requests per hour

@dataclass
class AggregatedData:
    """Represents aggregated data from multiple sources."""
    data_type: str
    value: Any
    confidence_score: float
    contributing_sources: List[str]
    timestamp: datetime
    ttl_seconds: int = 3600  # Time to live in cache

class CacheManager:
    """Manages caching for API responses."""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self.lock:
            if key in self.cache:
                item, expiry_time = self.cache[key]
                if datetime.now() < expiry_time:
                    self.access_times[key] = datetime.now()
                    return item
                else:
                    # Remove expired item
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set item in cache with TTL."""
        with self.lock:
            # Clean cache if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            expiry_time = datetime.now() + timedelta(seconds=ttl_seconds)
            self.cache[key] = (value, expiry_time)
            self.access_times[key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

class LoadBalancer:
    """Manages load balancing across multiple servers."""
    
    def __init__(self, servers: List[Dict[str, Any]]):
        """
        Initialize load balancer with server configurations.
        servers: List of dicts with 'id', 'endpoint', 'max_capacity', 'current_load', 'latency'
        """
        self.servers = {server['id']: server for server in servers}
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(deque)
        self.lock = threading.Lock()
    
    def calculate_server_load(self, server_id: str) -> float:
        """
        Calculate server load: Server_load(i) = Current_requests(i) / Max_capacity(i)
        """
        if server_id not in self.servers:
            return float('inf')
        
        server = self.servers[server_id]
        return server['current_load'] / server['max_capacity']
    
    def select_optimal_server(self) -> str:
        """
        Select optimal server using: argmin(Server_load(i) + Network_latency(i))
        """
        best_server = None
        best_score = float('inf')
        
        with self.lock:
            for server_id, server in self.servers.items():
                load = self.calculate_server_load(server_id)
                latency = server.get('latency', 0.1)  # Default 100ms latency
                
                score = load + latency
                
                if score < best_score:
                    best_score = score
                    best_server = server_id
        
        return best_server
    
    def update_server_load(self, server_id: str, load_change: int):
        """Update server load after request completion."""
        with self.lock:
            if server_id in self.servers:
                self.servers[server_id]['current_load'] += load_change
                # Ensure load doesn't go negative
                self.servers[server_id]['current_load'] = max(0, self.servers[server_id]['current_load'])
    
    def record_response_time(self, server_id: str, response_time: float):
        """Record response time for server performance tracking."""
        with self.lock:
            self.response_times[server_id].append(response_time)
            # Keep only last 100 response times
            if len(self.response_times[server_id]) > 100:
                self.response_times[server_id].popleft()
            
            # Update server latency based on recent performance
            if len(self.response_times[server_id]) > 0:
                avg_response_time = np.mean(list(self.response_times[server_id]))
                self.servers[server_id]['latency'] = avg_response_time

class APIIntegrationLayer:
    """Main API integration layer managing data aggregation and query processing."""
    
    def __init__(self, cache_size: int = 10000):
        """Initialize the API integration layer."""
        self.data_sources: Dict[str, DataSource] = {}
        self.cache_manager = CacheManager(max_size=cache_size)
        self.load_balancer = None
        self.external_platforms = {
            'google_maps': {'reliability': 0.9, 'weight': 0.4},
            'yelp': {'reliability': 0.85, 'weight': 0.3},
            'internal_validated': {'reliability': 0.95, 'weight': 0.3}
        }
        
        # Initialize load balancer with example servers
        servers = [
            {'id': 'server1', 'endpoint': 'http://api1.example.com', 'max_capacity': 100, 'current_load': 0, 'latency': 0.1},
            {'id': 'server2', 'endpoint': 'http://api2.example.com', 'max_capacity': 120, 'current_load': 0, 'latency': 0.12},
            {'id': 'server3', 'endpoint': 'http://api3.example.com', 'max_capacity': 80, 'current_load': 0, 'latency': 0.08}
        ]
        self.load_balancer = LoadBalancer(servers)
    
    def register_data_source(self, source_id: str, name: str, reliability_score: float, 
                           api_endpoint: Optional[str] = None):
        """Register a new external data source."""
        source = DataSource(
            source_id=source_id,
            name=name,
            reliability_score=reliability_score,
            last_updated=datetime.now(),
            api_endpoint=api_endpoint
        )
        self.data_sources[source_id] = source
    
    def calculate_freshness_factor(self, timestamp: datetime, decay_lambda: float = 0.1) -> float:
        """
        Calculate freshness factor: e^(-λ·age) where age is in hours.
        """
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return np.exp(-decay_lambda * age_hours)
    
    def calculate_aggregation_weights(self, data_sources: List[Dict]) -> List[float]:
        """
        Calculate weights for data aggregation:
        w_i = reliability_i · freshness_i / Σ(reliability_j · freshness_j)
        """
        weights = []
        total_weighted_score = 0
        
        # Calculate individual weighted scores
        for source_data in data_sources:
            reliability = source_data.get('reliability', 0.5)
            freshness = self.calculate_freshness_factor(source_data.get('timestamp', datetime.now()))
            weighted_score = reliability * freshness
            weights.append(weighted_score)
            total_weighted_score += weighted_score
        
        # Normalize weights
        if total_weighted_score > 0:
            weights = [w / total_weighted_score for w in weights]
        else:
            # Uniform weights if all scores are zero
            weights = [1.0 / len(data_sources)] * len(data_sources)
        
        return weights
    
    def aggregate_data(self, data_sources: List[Dict], data_type: str = "numeric") -> AggregatedData:
        """
        Aggregate data from multiple sources using weighted average:
        Aggregated_value = Σ(i=1 to n) w_i · value_i
        """
        if not data_sources:
            return None
        
        weights = self.calculate_aggregation_weights(data_sources)
        
        if data_type == "numeric":
            # Weighted average for numeric data
            values = [source.get('value', 0) for source in data_sources]
            aggregated_value = sum(w * v for w, v in zip(weights, values))
            
        elif data_type == "categorical":
            # Weighted voting for categorical data
            value_weights = defaultdict(float)
            for source, weight in zip(data_sources, weights):
                value = source.get('value', '')
                value_weights[value] += weight
            
            # Select category with highest weighted vote
            aggregated_value = max(value_weights.keys(), key=lambda k: value_weights[k])
            
        else:  # text or other types
            # For text, we could implement more sophisticated aggregation
            # For now, just take the value from the most reliable recent source
            most_reliable_idx = np.argmax(weights)
            aggregated_value = data_sources[most_reliable_idx].get('value', '')
        
        # Calculate confidence score based on agreement and weights
        confidence_score = self._calculate_confidence_score(data_sources, weights, aggregated_value)
        
        contributing_sources = [source.get('source_id', f'source_{i}') for i, source in enumerate(data_sources)]
        
        return AggregatedData(
            data_type=data_type,
            value=aggregated_value,
            confidence_score=confidence_score,
            contributing_sources=contributing_sources,
            timestamp=datetime.now()
        )
    
    def _calculate_confidence_score(self, data_sources: List[Dict], weights: List[float], 
                                  aggregated_value: Any) -> float:
        """Calculate confidence score based on source agreement and reliability."""
        if not data_sources:
            return 0.0
        
        agreement_score = 0.0
        total_weight = sum(weights)
        
        for source, weight in zip(data_sources, weights):
            source_value = source.get('value')
            
            # Calculate agreement based on data type
            if isinstance(aggregated_value, (int, float)) and isinstance(source_value, (int, float)):
                # For numeric values, use relative difference
                if aggregated_value != 0:
                    relative_diff = abs(source_value - aggregated_value) / abs(aggregated_value)
                    agreement = max(0, 1 - relative_diff)
                else:
                    agreement = 1.0 if source_value == 0 else 0.0
            else:
                # For categorical/text values, use exact match
                agreement = 1.0 if source_value == aggregated_value else 0.0
            
            agreement_score += weight * agreement
        
        # Normalize by total weight
        if total_weight > 0:
            agreement_score /= total_weight
        
        return agreement_score
    
    def calculate_cross_platform_consistency(self, platform_data: Dict[str, Any], 
                                           aggregated_data: Any, max_difference: float = 1.0) -> float:
        """
        Calculate cross-platform consistency:
        Consistency_score = 1 - Σ|platform_i_data - aggregated_data| / (n · max_possible_difference)
        """
        if not platform_data:
            return 1.0
        
        total_difference = 0.0
        n_platforms = len(platform_data)
        
        for platform, data in platform_data.items():
            if isinstance(data, (int, float)) and isinstance(aggregated_data, (int, float)):
                difference = abs(data - aggregated_data)
            else:
                # For non-numeric data, use binary difference (0 or 1)
                difference = 0.0 if data == aggregated_data else 1.0
            
            total_difference += difference
        
        # Calculate consistency score
        if n_platforms > 0 and max_difference > 0:
            consistency_score = 1 - (total_difference / (n_platforms * max_difference))
            return max(0.0, consistency_score)
        
        return 1.0
    
    def calculate_cross_validation_score(self, external_platforms: Dict[str, Any], 
                                       internal_data: Any) -> float:
        """
        Calculate cross-validation score:
        Cross_validation_score = Σ(i=1 to k) w_i · agreement_i
        """
        total_score = 0.0
        total_weight = 0.0
        
        for platform_id, platform_data in external_platforms.items():
            if platform_id in self.external_platforms:
                platform_config = self.external_platforms[platform_id]
                weight = platform_config['weight']
                reliability = platform_config['reliability']
                
                # Calculate agreement
                if isinstance(platform_data, (int, float)) and isinstance(internal_data, (int, float)):
                    if internal_data != 0:
                        agreement = 1 - min(1.0, abs(platform_data - internal_data) / abs(internal_data))
                    else:
                        agreement = 1.0 if platform_data == 0 else 0.0
                else:
                    agreement = 1.0 if platform_data == internal_data else 0.0
                
                # Weight by reliability
                weighted_agreement = weight * reliability * agreement
                total_score += weighted_agreement
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    @lru_cache(maxsize=1000)
    def calculate_query_response_time(self, cache_hit_ratio: float, cache_response_time: float = 0.01, 
                                    database_query_time: float = 0.15) -> float:
        """
        Calculate expected response time:
        Response_time = Cache_hit_ratio · Cache_response_time + (1 - Cache_hit_ratio) · Database_query_time
        """
        return cache_hit_ratio * cache_response_time + (1 - cache_hit_ratio) * database_query_time
    
    def process_consumer_query(self, query: str, query_params: Dict) -> Dict[str, Any]:
        """Process a consumer query with caching and load balancing."""
        # Generate cache key
        cache_key = hashlib.md5(f"{query}_{json.dumps(query_params, sort_keys=True)}".encode()).hexdigest()
        
        # Check cache first
        start_time = time.time()
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result:
            response_time = time.time() - start_time
            return {
                'data': cached_result,
                'response_time': response_time,
                'cache_hit': True,
                'server_used': None
            }
        
        # Cache miss - process query
        selected_server = self.load_balancer.select_optimal_server()
        
        # Update server load
        self.load_balancer.update_server_load(selected_server, 1)
        
        try:
            # Simulate query processing
            processed_data = self._execute_query(query, query_params)
            
            # Cache the result
            self.cache_manager.set(cache_key, processed_data)
            
            response_time = time.time() - start_time
            
            # Record server performance
            self.load_balancer.record_response_time(selected_server, response_time)
            
            return {
                'data': processed_data,
                'response_time': response_time,
                'cache_hit': False,
                'server_used': selected_server
            }
        
        finally:
            # Decrease server load
            self.load_balancer.update_server_load(selected_server, -1)
    
    def _execute_query(self, query: str, query_params: Dict) -> Dict[str, Any]:
        """Execute the actual query processing."""
        # Simulate data processing
        time.sleep(0.1)  


"""
This package contains the core functionality for running matches between agents
in the Lux AI Challenge Season 3. It includes utilities for running matches,
testing pathfinding algorithms, and implementing different agent strategies.
"""

from .agents import BestAgentBetterShooter
from submissions.best_agent_attacker import BestAgentAttacker

__all__ = ['BestAgentAttacker', 'BestAgentBetterShooter']

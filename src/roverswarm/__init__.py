"""RoverSwarm: exploration coopérative de robots sur grille."""

from .config import CommConfig, WorldConfig
from .world import RoverWorld

__all__ = ["CommConfig", "WorldConfig", "RoverWorld"]
__version__ = "1.0.0"

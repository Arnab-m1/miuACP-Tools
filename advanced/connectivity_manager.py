"""
Connectivity Manager for µACP

Manages intermittent connectivity for edge devices.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ConnectionState(Enum):
    """Connection states"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


@dataclass
class OfflineQueue:
    """Offline message queue"""
    messages: List[Dict[str, Any]]
    max_size: int
    timestamp: float


class ConnectivityManager:
    """Connectivity manager for µACP"""
    
    def __init__(self):
        self.connection_state = ConnectionState.CONNECTED
        self.offline_queue = OfflineQueue([], 100, time.time())
    
    def handle_disconnection(self):
        """Handle device disconnection"""
        self.connection_state = ConnectionState.DISCONNECTED
    
    def handle_reconnection(self):
        """Handle device reconnection"""
        self.connection_state = ConnectionState.CONNECTED

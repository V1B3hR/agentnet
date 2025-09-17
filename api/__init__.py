"""
AgentNet API module - Basic REST API for multi-agent sessions.
"""

try:
    from .server import AgentNetAPI
    from .models import SessionRequest, SessionResponse, AgentConfig
    API_AVAILABLE = True
    __all__ = ['AgentNetAPI', 'SessionRequest', 'SessionResponse', 'AgentConfig']
except ImportError:
    API_AVAILABLE = False
    __all__ = []
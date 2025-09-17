"""
Basic API server for AgentNet multi-agent orchestration.

This provides the P1 basic API foundation using a simple HTTP server
(since FastAPI is not available in the test environment).
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Import relative to parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from AgentNet import AgentNet, ExampleEngine
from .models import DialogueMode, AgentConfig

logger = logging.getLogger("agentnet.api")


class AgentNetAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for AgentNet API."""
    
    def __init__(self, api_server, *args, **kwargs):
        self.api_server = api_server
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == "/health":
                self._send_json_response({
                    "status": "healthy",
                    "timestamp": time.time(),
                    "active_sessions": len(self.api_server.sessions),
                    "version": "1.0.0"
                })
            elif path == "/sessions":
                sessions = [
                    {
                        "session_id": sid,
                        "status": data["status"],
                        "created_at": data["created_at"],
                        "agents": data["agents"]
                    }
                    for sid, data in self.api_server.sessions.items()
                ]
                self._send_json_response({"sessions": sessions})
            elif path.startswith("/sessions/") and path.endswith("/status"):
                session_id = path.split("/")[2]
                status = self.api_server.get_session_status(session_id)
                self._send_json_response(status)
            elif path.startswith("/sessions/"):
                session_id = path.split("/")[2]
                session = self.api_server.get_session(session_id)
                self._send_json_response(session)
            else:
                self._send_error_response(404, "Not found")
        except Exception as e:
            logger.error(f"GET error: {e}")
            self._send_error_response(500, str(e))
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            if path == "/sessions":
                response = self.api_server.create_session(request_data)
                self._send_json_response(response)
            elif path.startswith("/sessions/") and path.endswith("/run"):
                session_id = path.split("/")[2]
                response = asyncio.run(self.api_server.run_session(session_id))
                self._send_json_response(response)
            else:
                self._send_error_response(404, "Not found")
        except Exception as e:
            logger.error(f"POST error: {e}")
            self._send_error_response(500, str(e))
    
    def _send_json_response(self, data: dict, status_code: int = 200):
        """Send a JSON response."""
        response = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def _send_error_response(self, status_code: int, message: str):
        """Send an error response."""
        error_data = {"error": message, "status_code": status_code}
        self._send_json_response(error_data, status_code)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")


class AgentNetAPI:
    """Basic HTTP API for AgentNet multi-agent orchestration."""
    
    def __init__(self, engine=None):
        self.engine = engine or ExampleEngine()
        
        # Session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_agents: Dict[str, List[AgentNet]] = {}
    
    def create_session(self, request_data: dict) -> dict:
        """Create a new multi-agent dialogue session."""
        session_id = f"api_session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Create agents from configuration
        agents = []
        for agent_data in request_data["agents"]:
            agent = AgentNet(
                name=agent_data["name"],
                style=agent_data["style"],
                engine=self.engine,
                monitors=[]  # For now, no monitors in API
            )
            
            # Apply convergence configuration if provided
            if request_data.get("convergence_config"):
                agent.dialogue_config.update(request_data["convergence_config"])
            
            agents.append(agent)
        
        self.session_agents[session_id] = agents
        
        # Store session metadata
        session_data = {
            "session_id": session_id,
            "status": "ready",
            "topic": request_data["topic"],
            "mode": request_data.get("mode", "general"),
            "max_rounds": request_data.get("max_rounds", 5),
            "convergence": request_data.get("convergence", True),
            "parallel_round": request_data.get("parallel_round", False),
            "created_at": time.time(),
            "agents": [agent.name for agent in agents]
        }
        
        self.sessions[session_id] = session_data
        
        logger.info(f"Created session {session_id} with {len(agents)} agents")
        
        return {
            "session_id": session_id,
            "status": "ready",
            "participants": [agent.name for agent in agents]
        }
    
    def get_session(self, session_id: str) -> dict:
        """Get session status and results."""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session_data = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": session_data["status"],
            "topic_start": session_data.get("topic_start"),
            "topic_final": session_data.get("topic_final"),
            "converged": session_data.get("converged"),
            "rounds_executed": session_data.get("rounds_executed"),
            "participants": session_data.get("agents"),
            "transcript": session_data.get("transcript", [])[:10] if session_data.get("transcript") else []  # Limit transcript size
        }
    
    async def run_session(self, session_id: str) -> dict:
        """Run a complete multi-agent dialogue session."""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session_data = self.sessions[session_id]
        if session_data["status"] != "ready":
            raise ValueError(f"Session is {session_data['status']}, not ready")
        
        session_data["status"] = "running"
        agents = self.session_agents[session_id]
        
        # Run the dialogue
        result = await agents[0].async_multi_party_dialogue(
            agents=agents,
            topic=session_data["topic"],
            rounds=session_data["max_rounds"],
            mode=session_data["mode"],
            convergence=session_data["convergence"],
            parallel_round=session_data["parallel_round"]
        )
        
        # Update session with results
        session_data.update({
            "status": "completed",
            "topic_start": result.get("topic_start"),
            "topic_final": result.get("topic_final"),
            "converged": result.get("converged"),
            "rounds_executed": result.get("rounds_executed"),
            "transcript": result.get("transcript"),
            "final_summary": result.get("final_summary"),
            "completed_at": time.time()
        })
        
        logger.info(f"Session {session_id} completed: converged={result.get('converged')}, "
                  f"rounds={result.get('rounds_executed')}")
        
        return {
            "session_id": session_id,
            "status": "completed",
            "topic_start": result.get("topic_start"),
            "topic_final": result.get("topic_final"),
            "converged": result.get("converged"),
            "rounds_executed": result.get("rounds_executed"),
            "participants": result.get("participants")
        }
    
    def get_session_status(self, session_id: str) -> dict:
        """Get brief session status."""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session_data = self.sessions[session_id]
        transcript = session_data.get("transcript", [])
        current_round = len(transcript) // len(session_data["agents"]) if transcript else 0
        last_speaker = transcript[-1]["agent"] if transcript else None
        
        return {
            "session_id": session_id,
            "status": session_data["status"],
            "current_round": current_round,
            "total_rounds": session_data["max_rounds"],
            "converged": session_data.get("converged", False),
            "participants": session_data["agents"],
            "last_speaker": last_speaker
        }
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the API server."""
        def handler(*args, **kwargs):
            return AgentNetAPIHandler(self, *args, **kwargs)
        
        server = HTTPServer((host, port), handler)
        logger.info(f"AgentNet API server starting on {host}:{port}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("API server shutting down")
            server.shutdown()
            server.server_close()


# CLI entry point for running the API server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentNet API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    api = AgentNetAPI()
    api.run(host=args.host, port=args.port)
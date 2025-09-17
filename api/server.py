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

# P3: Import DAG and Evaluation components
from agentnet import (
    DAGPlanner, TaskNode, TaskGraph, TaskScheduler, ExecutionResult,
    EvaluationRunner, EvaluationScenario, EvaluationSuite, 
    MetricsCalculator, EvaluationMetrics, SuccessCriteria
)

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
            # Handle POST requests with or without body
            request_data = {}
            if 'Content-Length' in self.headers:
                content_length = int(self.headers['Content-Length'])
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
                    request_data = json.loads(post_data.decode('utf-8'))
            
            if path == "/sessions":
                response = self.api_server.create_session(request_data)
                self._send_json_response(response)
            elif path.startswith("/sessions/") and path.endswith("/run"):
                session_id = path.split("/")[2]
                response = asyncio.run(self.api_server.run_session(session_id))
                self._send_json_response(response)
            elif path == "/tasks/plan":
                response = self.api_server.plan_task_graph(request_data)
                self._send_json_response(response)
            elif path == "/tasks/execute":
                response = asyncio.run(self.api_server.execute_task_graph(request_data))
                self._send_json_response(response)
            elif path == "/eval/run":
                response = asyncio.run(self.api_server.run_evaluation(request_data))
                self._send_json_response(response)
            else:
                self._send_error_response(404, "Not found")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self._send_error_response(400, f"Invalid JSON: {e}")
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
        
        # P3: DAG and Evaluation components
        self.dag_planner = DAGPlanner()
        self.task_scheduler = TaskScheduler(max_retries=3, parallel_execution=True)
        self.evaluation_runner = EvaluationRunner(results_dir="api_eval_results")
        
        # Set up task executor for scheduler
        self.task_scheduler.set_task_executor(self._agentnet_task_executor)
        
        # Set up evaluation executors
        self.evaluation_runner.set_dialogue_executor(self._evaluation_dialogue_executor)
        self.evaluation_runner.set_workflow_executor(self._evaluation_workflow_executor)
    
    async def _agentnet_task_executor(self, task_id: str, prompt: str, agent_name: str, context: dict) -> dict:
        """Task executor that uses AgentNet instances."""
        # Create agent for this task
        agent = AgentNet(
            name=agent_name,
            style={"logic": 0.8, "creativity": 0.6, "analytical": 0.8},
            engine=self.engine
        )
        
        # Add context to prompt
        enhanced_prompt = prompt
        if context.get("dependency_results"):
            dep_summary = "\n".join([
                f"From {dep_id}: {result.get('content', 'No content')}" 
                for dep_id, result in context["dependency_results"].items()
            ])
            enhanced_prompt = f"{prompt}\n\nContext from dependencies:\n{dep_summary}"
        
        # Execute with AgentNet
        result = agent.generate_reasoning_tree(enhanced_prompt)
        
        return {
            "content": result["result"]["content"],
            "confidence": result["result"]["confidence"],
            "agent": agent_name,
            "task_id": task_id,
            "meta_insights": result["result"].get("meta_insights", [])
        }
    
    async def _evaluation_dialogue_executor(self, agents: List[str], topic: str, config: dict) -> dict:
        """Dialogue executor for evaluation scenarios."""
        # Create session for evaluation
        session_data = {
            "agents": [{"name": name, "style": {"logic": 0.8, "creativity": 0.6, "analytical": 0.8}} for name in agents],
            "topic": topic,
            "mode": config.get("mode", "general"),
            "max_rounds": config.get("max_rounds", 3),
            "convergence": config.get("convergence", True),
            "parallel_round": config.get("parallel_round", False)
        }
        
        # Create and run session
        session_result = self.create_session(session_data)
        session_id = session_result["session_id"]
        
        # Run the session
        result = await self.run_session(session_id)
        
        # Get full session data
        session_full = self.get_session(session_id)
        
        return {
            "transcript": session_full.get("transcript", []),
            "converged": session_full.get("converged", False),
            "rounds": session_full.get("rounds_executed", 0),
            "violations": []  # TODO: Add violations from monitors
        }
    
    async def _evaluation_workflow_executor(self, task_graph: dict, config: dict) -> dict:
        """Workflow executor for evaluation scenarios."""
        # Create TaskGraph from dict
        graph = self.dag_planner.create_graph_from_dict(task_graph)
        if not graph.is_valid:
            raise ValueError(f"Invalid task graph: {graph.validation_errors}")
        
        # Execute the graph
        result = await self.task_scheduler.execute_graph(graph, config)
        
        return {
            "status": result.status,
            "task_results": result.task_results,
            "execution_time": result.total_time
        }
    
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
    
    # P3: DAG & Evaluation endpoints
    
    def plan_task_graph(self, request_data: dict) -> dict:
        """POST /tasks/plan - Generate task DAG."""
        try:
            # Create task graph from request
            graph = self.dag_planner.create_graph_from_dict(request_data)
            
            if not graph.is_valid:
                return {
                    "error": "Invalid task graph",
                    "validation_errors": graph.validation_errors,
                    "graph_id": graph.graph_id
                }
            
            # Analyze the graph
            analysis = self.dag_planner.analyze_graph(graph)
            execution_order = self.dag_planner.get_execution_order(graph)
            
            return {
                "graph_id": graph.graph_id,
                "valid": graph.is_valid,
                "analysis": analysis,
                "execution_order": execution_order,
                "graph_data": graph.to_dict()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def execute_task_graph(self, request_data: dict) -> dict:
        """POST /tasks/execute - Execute DAG."""
        try:
            # Create task graph
            graph = self.dag_planner.create_graph_from_dict(request_data.get("task_graph", {}))
            
            if not graph.is_valid:
                return {
                    "error": "Invalid task graph",
                    "validation_errors": graph.validation_errors
                }
            
            # Execute the graph
            context = request_data.get("context", {})
            result = await self.task_scheduler.execute_graph(graph, context)
            
            return {
                "execution_id": result.execution_id,
                "status": result.status,
                "completed_tasks": result.get_completed_tasks(),
                "failed_tasks": result.get_failed_tasks(),
                "total_time": result.total_time,
                "task_results": result.to_dict()["task_results"]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def run_evaluation(self, request_data: dict) -> dict:
        """POST /eval/run - Trigger evaluation suite."""
        try:
            # Parse evaluation request
            if "suite" in request_data:
                # Run complete suite
                suite = EvaluationSuite.from_dict(request_data["suite"])
                result = await self.evaluation_runner.run_suite(
                    suite, 
                    context=request_data.get("context", {}),
                    parallel=request_data.get("parallel", False)
                )
                
                return {
                    "suite_name": result.suite_name,
                    "execution_id": result.execution_id,
                    "summary": result.summary,
                    "total_time": result.total_time,
                    "scenario_count": len(result.scenario_results)
                }
                
            elif "scenario" in request_data:
                # Run single scenario
                scenario = EvaluationScenario.from_dict(request_data["scenario"])
                result = await self.evaluation_runner.run_scenario(
                    scenario,
                    context=request_data.get("context", {})
                )
                
                return {
                    "scenario_name": result.scenario_name,
                    "execution_id": result.execution_id,
                    "status": result.status,
                    "metrics": result.metrics.to_dict() if result.metrics else None,
                    "execution_time": result.execution_time
                }
            else:
                return {"error": "Must specify either 'suite' or 'scenario'"}
                
        except Exception as e:
            return {"error": str(e)}
    
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
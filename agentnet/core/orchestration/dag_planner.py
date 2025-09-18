"""
DAG Planner for Task Graph Execution

Implements the DAG planner component that generates and validates directed acyclic graphs
for workflow automation. Based on FR7 requirements from docs/RoadmapAgentNet.md.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger("agentnet.orchestration.dag_planner")


@dataclass
class TaskNode:
    """
    Represents a task node in the DAG.
    
    Based on the roadmap example:
    {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []}
    """
    id: str
    prompt: str
    agent: str
    deps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the task node after initialization."""
        if not self.id:
            raise ValueError("Task node ID cannot be empty")
        if not self.prompt:
            raise ValueError("Task node prompt cannot be empty")
        if not self.agent:
            raise ValueError("Task node agent cannot be empty")


@dataclass
class TaskGraph:
    """
    Represents a complete task graph with nodes and validation status.
    """
    nodes: List[TaskNode]
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "graph_id": self.graph_id,
            "nodes": [
                {
                    "id": node.id,
                    "prompt": node.prompt,
                    "agent": node.agent,
                    "deps": node.deps,
                    "metadata": node.metadata
                }
                for node in self.nodes
            ],
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """Create TaskGraph from dictionary."""
        nodes = [
            TaskNode(
                id=node_data["id"],
                prompt=node_data["prompt"],
                agent=node_data["agent"],
                deps=node_data.get("deps", []),
                metadata=node_data.get("metadata", {})
            )
            for node_data in data["nodes"]
        ]
        
        graph = cls(
            nodes=nodes,
            metadata=data.get("metadata", {}),
            graph_id=data.get("graph_id", str(uuid.uuid4()))
        )
        graph.is_valid = data.get("is_valid", False)
        graph.validation_errors = data.get("validation_errors", [])
        return graph


class DAGPlanner:
    """
    DAG Planner generates and validates directed acyclic graphs for task execution.
    
    Key capabilities:
    - Generate DAG from task specifications
    - Validate DAG structure (no cycles, valid dependencies)
    - Optimize task ordering
    - Provide execution hints for scheduler
    """
    
    def __init__(self):
        self.logger = logger
    
    def create_graph_from_nodes(self, nodes: List[TaskNode]) -> TaskGraph:
        """
        Create a TaskGraph from a list of TaskNodes with validation.
        
        Args:
            nodes: List of TaskNode objects
            
        Returns:
            TaskGraph with validation results
        """
        task_graph = TaskGraph(nodes=nodes)
        self._validate_graph(task_graph)
        return task_graph
    
    def create_graph_from_json(self, json_data: str) -> TaskGraph:
        """
        Create a TaskGraph from JSON string.
        
        Expected format matches roadmap example:
        {
          "nodes": [
            {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
            {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]}
          ]
        }
        
        Args:
            json_data: JSON string containing task graph definition
            
        Returns:
            TaskGraph with validation results
        """
        try:
            data = json.loads(json_data)
            return self.create_graph_from_dict(data)
        except json.JSONDecodeError as e:
            task_graph = TaskGraph(nodes=[])
            task_graph.validation_errors.append(f"Invalid JSON: {str(e)}")
            return task_graph
    
    def create_graph_from_dict(self, data: Dict[str, Any]) -> TaskGraph:
        """
        Create a TaskGraph from dictionary.
        
        Args:
            data: Dictionary containing task graph definition
            
        Returns:
            TaskGraph with validation results
        """
        try:
            nodes = []
            for node_data in data.get("nodes", []):
                node = TaskNode(
                    id=node_data["id"],
                    prompt=node_data["prompt"],
                    agent=node_data["agent"],
                    deps=node_data.get("deps", []),
                    metadata=node_data.get("metadata", {})
                )
                nodes.append(node)
            
            task_graph = TaskGraph(
                nodes=nodes,
                metadata=data.get("metadata", {})
            )
            self._validate_graph(task_graph)
            return task_graph
            
        except (KeyError, ValueError) as e:
            task_graph = TaskGraph(nodes=[])
            task_graph.validation_errors.append(f"Invalid task graph structure: {str(e)}")
            return task_graph
    
    def _validate_graph(self, task_graph: TaskGraph) -> None:
        """
        Validate the task graph structure.
        
        Checks:
        1. No duplicate node IDs
        2. All dependencies exist as nodes
        3. No cycles in the dependency graph
        4. At least one root node (no dependencies)
        
        Args:
            task_graph: TaskGraph to validate (modified in place)
        """
        task_graph.validation_errors.clear()
        
        if not task_graph.nodes:
            task_graph.validation_errors.append("Task graph must contain at least one node")
            task_graph.is_valid = False
            return
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in task_graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [node_id for node_id in node_ids if node_ids.count(node_id) > 1]
            task_graph.validation_errors.append(f"Duplicate node IDs found: {set(duplicates)}")
        
        # Build node ID set for dependency validation
        node_id_set = set(node_ids)
        
        # Check that all dependencies exist
        for node in task_graph.nodes:
            for dep in node.deps:
                if dep not in node_id_set:
                    task_graph.validation_errors.append(
                        f"Node '{node.id}' depends on non-existent node '{dep}'"
                    )
        
        # Check for cycles using NetworkX
        try:
            nx_graph = self._build_networkx_graph(task_graph)
            if not nx.is_directed_acyclic_graph(nx_graph):
                cycles = list(nx.simple_cycles(nx_graph))
                task_graph.validation_errors.append(f"Cycles detected in graph: {cycles}")
        except Exception as e:
            task_graph.validation_errors.append(f"Error building graph for cycle detection: {str(e)}")
        
        # Check for at least one root node
        root_nodes = [node for node in task_graph.nodes if not node.deps]
        if not root_nodes:
            task_graph.validation_errors.append("Task graph must have at least one root node (node with no dependencies)")
        
        # Mark as valid if no errors
        task_graph.is_valid = len(task_graph.validation_errors) == 0
        
        if task_graph.is_valid:
            self.logger.info(f"Task graph validation successful: {len(task_graph.nodes)} nodes, {len(root_nodes)} root nodes")
        else:
            self.logger.warning(f"Task graph validation failed: {task_graph.validation_errors}")
    
    def _build_networkx_graph(self, task_graph: TaskGraph) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from TaskGraph for analysis.
        
        Args:
            task_graph: TaskGraph to convert
            
        Returns:
            NetworkX DiGraph
        """
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for node in task_graph.nodes:
            nx_graph.add_node(node.id, **{
                "prompt": node.prompt,
                "agent": node.agent,
                "metadata": node.metadata
            })
        
        # Add edges (dependencies)
        for node in task_graph.nodes:
            for dep in node.deps:
                nx_graph.add_edge(dep, node.id)
        
        return nx_graph
    
    def get_execution_order(self, task_graph: TaskGraph) -> List[List[str]]:
        """
        Get topological execution order for the task graph.
        
        Returns a list of lists, where each inner list contains node IDs
        that can be executed in parallel at that stage.
        
        Args:
            task_graph: Valid TaskGraph
            
        Returns:
            List of execution stages, each containing parallel-executable node IDs
            
        Raises:
            ValueError: If task graph is invalid
        """
        if not task_graph.is_valid:
            raise ValueError(f"Cannot get execution order for invalid task graph: {task_graph.validation_errors}")
        
        nx_graph = self._build_networkx_graph(task_graph)
        
        # Use topological generations to get parallel execution stages
        execution_stages = []
        for generation in nx.topological_generations(nx_graph):
            execution_stages.append(list(generation))
        
        self.logger.info(f"Generated execution order: {len(execution_stages)} stages")
        return execution_stages
    
    def get_ready_nodes(self, task_graph: TaskGraph, completed_nodes: Set[str]) -> List[str]:
        """
        Get nodes that are ready to execute given a set of completed nodes.
        
        Args:
            task_graph: Valid TaskGraph
            completed_nodes: Set of node IDs that have been completed
            
        Returns:
            List of node IDs ready for execution
            
        Raises:
            ValueError: If task graph is invalid
        """
        if not task_graph.is_valid:
            raise ValueError(f"Cannot get ready nodes for invalid task graph: {task_graph.validation_errors}")
        
        ready_nodes = []
        
        for node in task_graph.nodes:
            # Skip if already completed
            if node.id in completed_nodes:
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_nodes for dep in node.deps):
                ready_nodes.append(node.id)
        
        return ready_nodes
    
    def analyze_graph(self, task_graph: TaskGraph) -> Dict[str, Any]:
        """
        Analyze the task graph and return structural information.
        
        Args:
            task_graph: TaskGraph to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not task_graph.is_valid:
            return {
                "valid": False,
                "errors": task_graph.validation_errors,
                "node_count": len(task_graph.nodes)
            }
        
        nx_graph = self._build_networkx_graph(task_graph)
        
        # Find root and leaf nodes
        root_nodes = [node.id for node in task_graph.nodes if not node.deps]
        leaf_nodes = [node_id for node_id in nx_graph.nodes() if nx_graph.out_degree(node_id) == 0]
        
        # Calculate graph metrics
        analysis = {
            "valid": True,
            "node_count": len(task_graph.nodes),
            "edge_count": nx_graph.number_of_edges(),
            "root_nodes": root_nodes,
            "leaf_nodes": leaf_nodes,
            "max_depth": 0,
            "agents_involved": list(set(node.agent for node in task_graph.nodes)),
            "execution_stages": len(self.get_execution_order(task_graph))
        }
        
        # Calculate maximum depth
        if root_nodes:
            try:
                max_depth = 0
                for root in root_nodes:
                    for leaf in leaf_nodes:
                        if nx.has_path(nx_graph, root, leaf):
                            path_length = nx.shortest_path_length(nx_graph, root, leaf)
                            max_depth = max(max_depth, path_length)
                analysis["max_depth"] = max_depth
            except Exception as e:
                self.logger.warning(f"Could not calculate max depth: {str(e)}")
        
        return analysis


# Example usage and testing
if __name__ == "__main__":
    # Example from roadmap
    example_json = """
    {
      "nodes": [
        {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
        {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]},
        {"id": "mitigations", "prompt": "Propose mitigations", "agent": "Apollo", "deps": ["analysis"]},
        {"id": "summary", "prompt": "Integrate plan & mitigations", "agent": "Synthesizer", "deps": ["mitigations"]}
      ]
    }
    """
    
    planner = DAGPlanner()
    task_graph = planner.create_graph_from_json(example_json)
    
    print("Task Graph:")
    print(f"Valid: {task_graph.is_valid}")
    if task_graph.validation_errors:
        print(f"Errors: {task_graph.validation_errors}")
    
    if task_graph.is_valid:
        print(f"Execution order: {planner.get_execution_order(task_graph)}")
        print(f"Analysis: {planner.analyze_graph(task_graph)}")
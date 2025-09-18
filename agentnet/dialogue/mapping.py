"""
Dialogue mapping and visualization for collaborative decision-making.

Provides tools for visualizing dialogue structure, decision points, and
solution-building processes in multi-agent conversations.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .enhanced_modes import DialogueState, DialogueTurn

logger = logging.getLogger("agentnet.dialogue.mapping")


class NodeType(str, Enum):
    """Types of nodes in dialogue map."""
    TOPIC = "topic"
    STATEMENT = "statement"
    QUESTION = "question"
    ARGUMENT = "argument"
    SOLUTION = "solution"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    CRITERION = "criterion"


class EdgeType(str, Enum):
    """Types of edges in dialogue map."""
    RELATES_TO = "relates_to"
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    BUILDS_ON = "builds_on"
    ANSWERS = "answers"
    CLARIFIES = "clarifies"
    LEADS_TO = "leads_to"


@dataclass
class MapNode:
    """Node in the dialogue map."""
    id: str
    label: str
    node_type: NodeType
    agent: str
    round_number: int
    timestamp: float
    confidence: float = 0.5
    content: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MapEdge:
    """Edge in the dialogue map."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    label: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VisualizationConfig:
    """Configuration for dialogue map visualization."""
    layout: str = "force_directed"  # force_directed, hierarchical, circular
    node_size_range: Tuple[int, int] = (10, 30)
    edge_width_range: Tuple[int, int] = (1, 5)
    color_scheme: str = "categorical"  # categorical, confidence_based, agent_based
    show_labels: bool = True
    show_timestamps: bool = False
    group_by_agent: bool = True
    highlight_critical_path: bool = True


class DialogueMapper:
    """
    Maps dialogue structure and creates visualizations.
    
    Converts dialogue transcripts into graph structures that can be
    visualized and analyzed for patterns and insights.
    """
    
    def __init__(self):
        """Initialize dialogue mapper."""
        self.maps: Dict[str, Dict[str, Any]] = {}
        self.node_counter = 0
    
    def create_dialogue_map(self,
                          dialogue_state: 'DialogueState',
                          transcript: List['DialogueTurn'] = None,
                          mapping_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a dialogue map from dialogue state and transcript.
        
        Args:
            dialogue_state: Dialogue state object
            transcript: Optional transcript (uses state.transcript if None)
            mapping_config: Optional configuration for mapping behavior
            
        Returns:
            Dialogue map with nodes, edges, and metadata
        """
        if transcript is None:
            transcript = dialogue_state.transcript
        
        mapping_config = mapping_config or {}
        map_id = dialogue_state.session_id
        
        logger.info(f"Creating dialogue map: {map_id}")
        
        # Initialize map structure
        dialogue_map = {
            "id": map_id,
            "topic": dialogue_state.topic,
            "mode": dialogue_state.mode.value,
            "participants": dialogue_state.participants,
            "created_at": time.time(),
            "nodes": [],
            "edges": [],
            "metadata": {
                "total_rounds": dialogue_state.current_round,
                "participant_count": len(dialogue_state.participants),
                "mapping_config": mapping_config
            }
        }
        
        nodes = []
        edges = []
        
        # Create central topic node
        topic_node = MapNode(
            id="topic_0",
            label=dialogue_state.topic,
            node_type=NodeType.TOPIC,
            agent="system",
            round_number=0,
            timestamp=dialogue_state.transcript[0].timestamp if transcript else time.time(),
            confidence=1.0,
            content=dialogue_state.topic,
            metadata={"is_root": True}
        )
        nodes.append(topic_node)
        
        # Process each turn in the transcript
        previous_node_id = "topic_0"
        
        for turn in transcript:
            # Create node for this turn
            node_id = f"turn_{turn.round_number}_{turn.agent_name}"
            turn_node = self._create_turn_node(turn, node_id)
            nodes.append(turn_node)
            
            # Create edge from previous node
            edge = MapEdge(
                source=previous_node_id,
                target=node_id,
                edge_type=self._determine_edge_type(turn, nodes),
                weight=turn.confidence,
                label=f"R{turn.round_number}",
                metadata={"round": turn.round_number, "agent": turn.agent_name}
            )
            edges.append(edge)
            
            # Identify and create specialized nodes within the turn
            specialized_nodes, specialized_edges = self._extract_specialized_nodes(turn, turn_node)
            nodes.extend(specialized_nodes)
            edges.extend(specialized_edges)
            
            previous_node_id = node_id
        
        # Create cross-turn relationships
        cross_edges = self._identify_cross_turn_relationships(nodes, transcript)
        edges.extend(cross_edges)
        
        # Convert to serializable format
        dialogue_map["nodes"] = [asdict(node) for node in nodes]
        dialogue_map["edges"] = [asdict(edge) for edge in edges]
        
        # Store map
        self.maps[map_id] = dialogue_map
        
        return dialogue_map
    
    def _create_turn_node(self, turn: 'DialogueTurn', node_id: str) -> MapNode:
        """Create a map node from a dialogue turn."""
        # Determine node type based on content
        content_lower = turn.content.lower()
        
        if "?" in turn.content:
            node_type = NodeType.QUESTION
        elif any(word in content_lower for word in ["solution", "approach", "plan", "strategy"]):
            node_type = NodeType.SOLUTION
        elif any(word in content_lower for word in ["decision", "choose", "decide"]):
            node_type = NodeType.DECISION
        elif any(word in content_lower for word in ["argue", "disagree", "oppose", "support"]):
            node_type = NodeType.ARGUMENT
        else:
            node_type = NodeType.STATEMENT
        
        # Create appropriate label
        label = self._generate_node_label(turn.content, node_type)
        
        return MapNode(
            id=node_id,
            label=label,
            node_type=node_type,
            agent=turn.agent_name,
            round_number=turn.round_number,
            timestamp=turn.timestamp,
            confidence=turn.confidence,
            content=turn.content,
            metadata={
                "intensity": getattr(turn, 'intensity_level', None),
                "reasoning_type": getattr(turn, 'reasoning_type', None),
                "context_additions": getattr(turn, 'context_additions', None)
            }
        )
    
    def _generate_node_label(self, content: str, node_type: NodeType) -> str:
        """Generate appropriate label for a node."""
        # Truncate content for label
        max_length = 50
        truncated = content[:max_length] + "..." if len(content) > max_length else content
        
        # Add type prefix
        type_prefixes = {
            NodeType.QUESTION: "Q: ",
            NodeType.SOLUTION: "S: ",
            NodeType.DECISION: "D: ",
            NodeType.ARGUMENT: "A: ",
            NodeType.STATEMENT: ""
        }
        
        prefix = type_prefixes.get(node_type, "")
        return f"{prefix}{truncated}"
    
    def _determine_edge_type(self, turn: 'DialogueTurn', existing_nodes: List[MapNode]) -> EdgeType:
        """Determine the relationship type for this turn."""
        content_lower = turn.content.lower()
        
        # Check for specific relationship indicators
        if any(word in content_lower for word in ["agree", "support", "build on", "yes"]):
            return EdgeType.SUPPORTS
        elif any(word in content_lower for word in ["disagree", "oppose", "however", "but"]):
            return EdgeType.OPPOSES
        elif any(word in content_lower for word in ["answer", "response", "reply"]):
            return EdgeType.ANSWERS
        elif any(word in content_lower for word in ["clarify", "explain", "mean"]):
            return EdgeType.CLARIFIES
        elif any(word in content_lower for word in ["therefore", "thus", "leads to"]):
            return EdgeType.LEADS_TO
        else:
            return EdgeType.RELATES_TO
    
    def _extract_specialized_nodes(self, turn: 'DialogueTurn', parent_node: MapNode) -> Tuple[List[MapNode], List[MapEdge]]:
        """Extract specialized nodes from within a turn (e.g., constraints, criteria)."""
        nodes = []
        edges = []
        content_lower = turn.content.lower()
        
        # Extract constraints
        if any(word in content_lower for word in ["constraint", "limitation", "cannot", "must not"]):
            constraint_id = f"{parent_node.id}_constraint"
            constraint_node = MapNode(
                id=constraint_id,
                label=f"Constraint: {turn.content[:30]}...",
                node_type=NodeType.CONSTRAINT,
                agent=turn.agent_name,
                round_number=turn.round_number,
                timestamp=turn.timestamp,
                confidence=turn.confidence * 0.8,  # Slightly lower confidence for extracted elements
                content=turn.content,
                metadata={"parent_turn": parent_node.id}
            )
            nodes.append(constraint_node)
            
            # Create edge from parent
            edge = MapEdge(
                source=parent_node.id,
                target=constraint_id,
                edge_type=EdgeType.RELATES_TO,
                weight=0.8,
                label="identifies"
            )
            edges.append(edge)
        
        # Extract criteria
        if any(word in content_lower for word in ["criteria", "requirement", "important", "priority"]):
            criteria_id = f"{parent_node.id}_criteria"
            criteria_node = MapNode(
                id=criteria_id,
                label=f"Criteria: {turn.content[:30]}...",
                node_type=NodeType.CRITERION,
                agent=turn.agent_name,
                round_number=turn.round_number,
                timestamp=turn.timestamp,
                confidence=turn.confidence * 0.8,
                content=turn.content,
                metadata={"parent_turn": parent_node.id}
            )
            nodes.append(criteria_node)
            
            # Create edge from parent
            edge = MapEdge(
                source=parent_node.id,
                target=criteria_id,
                edge_type=EdgeType.RELATES_TO,
                weight=0.8,
                label="establishes"
            )
            edges.append(edge)
        
        return nodes, edges
    
    def _identify_cross_turn_relationships(self, nodes: List[MapNode], transcript: List['DialogueTurn']) -> List[MapEdge]:
        """Identify relationships that span across turns."""
        cross_edges = []
        
        # Simple implementation: look for question-answer pairs
        questions = [node for node in nodes if node.node_type == NodeType.QUESTION]
        statements = [node for node in nodes if node.node_type == NodeType.STATEMENT]
        
        for question in questions:
            # Look for statements in subsequent rounds from different agents
            for statement in statements:
                if (statement.round_number > question.round_number and 
                    statement.agent != question.agent and
                    statement.round_number <= question.round_number + 2):  # Within 2 rounds
                    
                    # Check if statement might be answering the question
                    # This is a simple heuristic - could be enhanced with NLP
                    cross_edge = MapEdge(
                        source=question.id,
                        target=statement.id,
                        edge_type=EdgeType.ANSWERS,
                        weight=0.6,  # Lower confidence for inferred relationships
                        label="answered by",
                        metadata={"inferred": True}
                    )
                    cross_edges.append(cross_edge)
                    break  # Only connect to first potential answer
        
        return cross_edges
    
    def generate_visualization_data(self,
                                  map_id: str,
                                  config: Optional[VisualizationConfig] = None) -> Dict[str, Any]:
        """
        Generate visualization data for a dialogue map.
        
        Args:
            map_id: ID of the dialogue map to visualize
            config: Optional visualization configuration
            
        Returns:
            Visualization data structure
        """
        if map_id not in self.maps:
            raise ValueError(f"Dialogue map not found: {map_id}")
        
        dialogue_map = self.maps[map_id]
        config = config or VisualizationConfig()
        
        # Prepare nodes for visualization
        vis_nodes = []
        for node_data in dialogue_map["nodes"]:
            vis_node = self._prepare_node_for_visualization(node_data, config)
            vis_nodes.append(vis_node)
        
        # Prepare edges for visualization
        vis_edges = []
        for edge_data in dialogue_map["edges"]:
            vis_edge = self._prepare_edge_for_visualization(edge_data, config)
            vis_edges.append(vis_edge)
        
        # Generate layout hints
        layout_config = self._generate_layout_config(dialogue_map, config)
        
        # Create legend
        legend = self._create_visualization_legend(config)
        
        visualization_data = {
            "map_id": map_id,
            "nodes": vis_nodes,
            "edges": vis_edges,
            "layout": {
                "type": config.layout,
                "config": layout_config
            },
            "legend": legend,
            "metadata": {
                "node_count": len(vis_nodes),
                "edge_count": len(vis_edges),
                "config": asdict(config)
            }
        }
        
        return visualization_data
    
    def _prepare_node_for_visualization(self, node_data: Dict[str, Any], config: VisualizationConfig) -> Dict[str, Any]:
        """Prepare a node for visualization."""
        node_type = NodeType(node_data["node_type"])
        
        # Determine node size based on confidence
        min_size, max_size = config.node_size_range
        size = min_size + (max_size - min_size) * node_data["confidence"]
        
        # Determine node color
        color = self._get_node_color(node_data, config)
        
        # Prepare visualization node
        vis_node = {
            "id": node_data["id"],
            "label": node_data["label"] if config.show_labels else "",
            "size": size,
            "color": color,
            "type": node_type.value,
            "agent": node_data["agent"],
            "round": node_data["round_number"],
            "confidence": node_data["confidence"]
        }
        
        # Add timestamp if requested
        if config.show_timestamps:
            vis_node["timestamp"] = node_data["timestamp"]
        
        # Add metadata for tooltips
        vis_node["metadata"] = {
            "content_preview": node_data["content"][:100] + "..." if len(node_data["content"]) > 100 else node_data["content"],
            "full_content": node_data["content"],
            "node_type": node_type.value,
            "agent": node_data["agent"],
            "round": node_data["round_number"],
            "confidence": node_data["confidence"]
        }
        
        return vis_node
    
    def _prepare_edge_for_visualization(self, edge_data: Dict[str, Any], config: VisualizationConfig) -> Dict[str, Any]:
        """Prepare an edge for visualization."""
        edge_type = EdgeType(edge_data["edge_type"])
        
        # Determine edge width based on weight
        min_width, max_width = config.edge_width_range
        width = min_width + (max_width - min_width) * edge_data["weight"]
        
        # Determine edge color based on type
        edge_colors = {
            EdgeType.SUPPORTS: "#4CAF50",      # Green
            EdgeType.OPPOSES: "#F44336",       # Red
            EdgeType.BUILDS_ON: "#2196F3",     # Blue
            EdgeType.ANSWERS: "#FF9800",       # Orange
            EdgeType.CLARIFIES: "#9C27B0",     # Purple
            EdgeType.LEADS_TO: "#607D8B",      # Blue Grey
            EdgeType.RELATES_TO: "#757575"     # Grey
        }
        
        color = edge_colors.get(edge_type, "#757575")
        
        # Prepare visualization edge
        vis_edge = {
            "source": edge_data["source"],
            "target": edge_data["target"],
            "width": width,
            "color": color,
            "type": edge_type.value,
            "label": edge_data["label"],
            "weight": edge_data["weight"]
        }
        
        # Add arrow for directed relationships
        if edge_type in [EdgeType.LEADS_TO, EdgeType.ANSWERS, EdgeType.CLARIFIES]:
            vis_edge["arrow"] = "target"
        
        return vis_edge
    
    def _get_node_color(self, node_data: Dict[str, Any], config: VisualizationConfig) -> str:
        """Determine node color based on configuration."""
        node_type = NodeType(node_data["node_type"])
        
        if config.color_scheme == "categorical":
            # Color by node type
            type_colors = {
                NodeType.TOPIC: "#4CAF50",        # Green
                NodeType.STATEMENT: "#2196F3",    # Blue
                NodeType.QUESTION: "#FF9800",     # Orange
                NodeType.ARGUMENT: "#F44336",     # Red
                NodeType.SOLUTION: "#9C27B0",     # Purple
                NodeType.DECISION: "#E91E63",     # Pink
                NodeType.CONSTRAINT: "#795548",   # Brown
                NodeType.CRITERION: "#607D8B"     # Blue Grey
            }
            return type_colors.get(node_type, "#757575")
        
        elif config.color_scheme == "confidence_based":
            # Color by confidence level
            confidence = node_data["confidence"]
            if confidence > 0.8:
                return "#4CAF50"  # High confidence - Green
            elif confidence > 0.6:
                return "#FF9800"  # Medium confidence - Orange
            else:
                return "#F44336"  # Low confidence - Red
        
        elif config.color_scheme == "agent_based":
            # Color by agent (would need agent color mapping)
            agent_colors = {
                "Agent1": "#4CAF50",
                "Agent2": "#2196F3", 
                "Agent3": "#FF9800",
                "Agent4": "#9C27B0"
            }
            return agent_colors.get(node_data["agent"], "#757575")
        
        else:
            return "#2196F3"  # Default blue
    
    def _generate_layout_config(self, dialogue_map: Dict[str, Any], config: VisualizationConfig) -> Dict[str, Any]:
        """Generate layout configuration for visualization."""
        layout_configs = {
            "force_directed": {
                "physics": True,
                "repulsion": 200,
                "spring_length": 100,
                "spring_strength": 0.1,
                "damping": 0.1
            },
            "hierarchical": {
                "direction": "UD",  # Up-Down
                "sort_method": "directed",
                "level_separation": 150,
                "node_spacing": 100
            },
            "circular": {
                "radius": 200,
                "start_angle": 0,
                "clockwise": True
            }
        }
        
        base_config = layout_configs.get(config.layout, layout_configs["force_directed"])
        
        # Add grouping configuration
        if config.group_by_agent:
            base_config["clustering"] = {
                "enabled": True,
                "cluster_by": "agent"
            }
        
        return base_config
    
    def _create_visualization_legend(self, config: VisualizationConfig) -> Dict[str, Any]:
        """Create legend for the visualization."""
        legend = {
            "node_types": {
                "topic": {"color": "#4CAF50", "description": "Main topic"},
                "statement": {"color": "#2196F3", "description": "Statement"},
                "question": {"color": "#FF9800", "description": "Question"},
                "argument": {"color": "#F44336", "description": "Argument"},
                "solution": {"color": "#9C27B0", "description": "Solution"},
                "decision": {"color": "#E91E63", "description": "Decision"},
                "constraint": {"color": "#795548", "description": "Constraint"},
                "criterion": {"color": "#607D8B", "description": "Criterion"}
            },
            "edge_types": {
                "supports": {"color": "#4CAF50", "description": "Supports"},
                "opposes": {"color": "#F44336", "description": "Opposes"},
                "builds_on": {"color": "#2196F3", "description": "Builds on"},
                "answers": {"color": "#FF9800", "description": "Answers"},
                "clarifies": {"color": "#9C27B0", "description": "Clarifies"},
                "leads_to": {"color": "#607D8B", "description": "Leads to"},
                "relates_to": {"color": "#757575", "description": "Relates to"}
            }
        }
        
        return legend
    
    def export_map(self, map_id: str, format: str = "json", file_path: Optional[str] = None) -> str:
        """
        Export a dialogue map to file.
        
        Args:
            map_id: ID of the map to export
            format: Export format (json, graphml, dot)
            file_path: Optional file path (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if map_id not in self.maps:
            raise ValueError(f"Dialogue map not found: {map_id}")
        
        dialogue_map = self.maps[map_id]
        
        if file_path is None:
            timestamp = int(time.time())
            file_path = f"dialogue_map_{map_id}_{timestamp}.{format}"
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(dialogue_map, f, indent=2, default=str)
        
        elif format == "graphml":
            # Export to GraphML format for use with graph analysis tools
            graphml_content = self._export_to_graphml(dialogue_map)
            with open(file_path, 'w') as f:
                f.write(graphml_content)
        
        elif format == "dot":
            # Export to DOT format for Graphviz
            dot_content = self._export_to_dot(dialogue_map)
            with open(file_path, 'w') as f:
                f.write(dot_content)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported dialogue map {map_id} to {file_path}")
        return file_path
    
    def _export_to_graphml(self, dialogue_map: Dict[str, Any]) -> str:
        """Export dialogue map to GraphML format."""
        # This is a simplified GraphML export - could be enhanced
        graphml_content = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="label" for="node" attr.name="label" attr.type="string"/>
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <key id="agent" for="node" attr.name="agent" attr.type="string"/>
  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>
  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>
  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="directed">
"""
        
        # Add nodes
        for node in dialogue_map["nodes"]:
            graphml_content += f"""    <node id="{node['id']}">
      <data key="label">{node['label']}</data>
      <data key="type">{node['node_type']}</data>
      <data key="agent">{node['agent']}</data>
      <data key="confidence">{node['confidence']}</data>
    </node>
"""
        
        # Add edges
        for edge in dialogue_map["edges"]:
            graphml_content += f"""    <edge source="{edge['source']}" target="{edge['target']}">
      <data key="relation">{edge['edge_type']}</data>
      <data key="weight">{edge['weight']}</data>
    </edge>
"""
        
        graphml_content += """  </graph>
</graphml>"""
        
        return graphml_content
    
    def _export_to_dot(self, dialogue_map: Dict[str, Any]) -> str:
        """Export dialogue map to DOT format for Graphviz."""
        dot_content = f'digraph "{dialogue_map["id"]}" {{\n'
        dot_content += f'  label="{dialogue_map["topic"]}"\n'
        dot_content += '  rankdir=TB\n\n'
        
        # Add nodes
        for node in dialogue_map["nodes"]:
            node_attrs = [
                f'label="{node["label"]}"',
                f'shape="box"',
                f'style="filled"'
            ]
            
            # Color based on node type
            type_colors = {
                "topic": "lightgreen",
                "statement": "lightblue",
                "question": "orange",
                "argument": "lightcoral",
                "solution": "plum",
                "decision": "pink"
            }
            color = type_colors.get(node["node_type"], "lightgray")
            node_attrs.append(f'fillcolor="{color}"')
            
            dot_content += f'  "{node["id"]}" [{", ".join(node_attrs)}]\n'
        
        dot_content += '\n'
        
        # Add edges
        for edge in dialogue_map["edges"]:
            edge_attrs = []
            
            if edge["label"]:
                edge_attrs.append(f'label="{edge["label"]}"')
            
            # Style based on edge type
            if edge["edge_type"] == "supports":
                edge_attrs.append('color="green"')
            elif edge["edge_type"] == "opposes":
                edge_attrs.append('color="red"')
            elif edge["edge_type"] == "answers":
                edge_attrs.append('color="blue"')
            
            attrs_str = f' [{", ".join(edge_attrs)}]' if edge_attrs else ''
            dot_content += f'  "{edge["source"]}" -> "{edge["target"]}"{attrs_str}\n'
        
        dot_content += '}\n'
        
        return dot_content
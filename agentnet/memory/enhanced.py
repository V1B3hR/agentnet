"""
Phase 7 Enhanced Memory Systems.

Implements next-generation memory capabilities:
- Episodic memory with temporal reasoning
- Hierarchical knowledge organization
- Cross-modal memory linking (text, code, data)
- Memory consolidation and forgetting mechanisms
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta

from .base import MemoryEntry, MemoryLayer, MemoryType, MemoryRetrieval

logger = logging.getLogger("agentnet.memory.enhanced")


class ModalityType(str, Enum):
    """Types of modalities for cross-modal memory."""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    STRUCTURED = "structured"
    TEMPORAL = "temporal"
    CONCEPTUAL = "conceptual"


class ConsolidationStrategy(str, Enum):
    """Memory consolidation strategies."""
    FREQUENCY_BASED = "frequency_based"
    RECENCY_BASED = "recency_based"
    IMPORTANCE_BASED = "importance_based"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TEMPORAL_CLUSTERING = "temporal_clustering"


@dataclass
class CrossModalLink:
    """A link between memories of different modalities."""
    
    source_id: str
    target_id: str
    source_modality: ModalityType
    target_modality: ModalityType
    relation_type: str
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCluster:
    """A cluster of related memories."""
    
    cluster_id: str
    memories: List[str]  # Memory IDs
    centroid: Optional[Dict[str, Any]] = None
    coherence_score: float = 0.0
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationRule:
    """A rule for memory consolidation."""
    
    name: str
    conditions: List[str]
    actions: List[str]
    priority: float = 1.0
    strategy: ConsolidationStrategy = ConsolidationStrategy.FREQUENCY_BASED


class HierarchicalKnowledgeOrganizer:
    """Organizes memories in hierarchical structures."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.hierarchy: Dict[str, Dict] = {}
        self.level_mappings: Dict[int, Set[str]] = {i: set() for i in range(max_depth)}
    
    def organize_memories(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Organize memories into hierarchical structure."""
        if not memories:
            return {}
        
        # Group memories by conceptual similarity
        concept_groups = self._group_by_concepts(memories)
        
        # Build hierarchy levels
        hierarchy = {}
        for level in range(self.max_depth):
            level_groups = self._create_level_groups(concept_groups, level)
            hierarchy[f"level_{level}"] = level_groups
            self.level_mappings[level].update(level_groups.keys())
        
        return hierarchy
    
    def _group_by_concepts(self, memories: List[MemoryEntry]) -> Dict[str, List[MemoryEntry]]:
        """Group memories by conceptual similarity."""
        groups = {}
        
        for memory in memories:
            # Extract key concepts (simplified)
            concepts = self._extract_concepts(memory.content)
            
            for concept in concepts:
                if concept not in groups:
                    groups[concept] = []
                groups[concept].append(memory)
        
        return groups
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        # Simplified concept extraction
        words = content.lower().split()
        # Filter for meaningful concepts (length > 3, alphabetic)
        concepts = [w for w in words if len(w) > 3 and w.isalpha()]
        return concepts[:5]  # Limit to top 5 concepts
    
    def _create_level_groups(self, concept_groups: Dict[str, List[MemoryEntry]], level: int) -> Dict[str, Any]:
        """Create groups for a specific hierarchy level."""
        if level == 0:
            # Base level: individual concepts
            return {concept: [m.content for m in memories] 
                   for concept, memories in concept_groups.items()}
        else:
            # Higher levels: group concepts together
            grouped = {}
            concept_keys = list(concept_groups.keys())
            
            # Group concepts in pairs/triples for higher levels
            group_size = min(level + 1, 3)
            for i in range(0, len(concept_keys), group_size):
                group_concepts = concept_keys[i:i + group_size]
                group_key = "_".join(group_concepts)
                
                group_memories = []
                for concept in group_concepts:
                    group_memories.extend(concept_groups[concept])
                
                grouped[group_key] = {
                    "concepts": group_concepts,
                    "memory_count": len(group_memories),
                    "level": level
                }
            
            return grouped
    
    def get_level_structure(self, level: int) -> Dict[str, Any]:
        """Get the structure at a specific hierarchy level."""
        if level < 0 or level >= self.max_depth:
            return {}
        
        return {
            "level": level,
            "groups": list(self.level_mappings[level]),
            "group_count": len(self.level_mappings[level])
        }


class CrossModalMemoryLinker:
    """Links memories across different modalities."""
    
    def __init__(self):
        self.links: List[CrossModalLink] = []
        self.modality_indices: Dict[ModalityType, Set[str]] = {
            modality: set() for modality in ModalityType
        }
    
    def add_link(self, link: CrossModalLink) -> None:
        """Add a cross-modal link."""
        self.links.append(link)
        self.modality_indices[link.source_modality].add(link.source_id)
        self.modality_indices[link.target_modality].add(link.target_id)
    
    def find_links(self, memory_id: str, modality: Optional[ModalityType] = None) -> List[CrossModalLink]:
        """Find links for a memory across modalities."""
        links = []
        
        for link in self.links:
            if link.source_id == memory_id or link.target_id == memory_id:
                if modality is None or link.source_modality == modality or link.target_modality == modality:
                    links.append(link)
        
        return links
    
    def create_automatic_links(self, memories: List[MemoryEntry]) -> List[CrossModalLink]:
        """Automatically create cross-modal links based on content similarity."""
        new_links = []
        
        # Group memories by modality
        modality_groups = self._group_by_modality(memories)
        
        # Create links between modalities
        modalities = list(modality_groups.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                modality1, modality2 = modalities[i], modalities[j]
                
                links = self._find_cross_modal_similarities(
                    modality_groups[modality1],
                    modality_groups[modality2],
                    modality1,
                    modality2
                )
                new_links.extend(links)
        
        # Add new links
        for link in new_links:
            self.add_link(link)
        
        return new_links
    
    def _group_by_modality(self, memories: List[MemoryEntry]) -> Dict[ModalityType, List[MemoryEntry]]:
        """Group memories by modality type."""
        groups = {modality: [] for modality in ModalityType}
        
        for memory in memories:
            modality = self._infer_modality(memory)
            groups[modality].append(memory)
        
        return groups
    
    def _infer_modality(self, memory: MemoryEntry) -> ModalityType:
        """Infer the modality type of a memory."""
        content = memory.content.lower()
        
        # Code indicators
        if any(keyword in content for keyword in ["def ", "class ", "import ", "function", "{", "}"]):
            return ModalityType.CODE
        
        # Data indicators
        if any(keyword in content for keyword in ["data", "dataset", "table", "json", "csv"]):
            return ModalityType.DATA
        
        # Temporal indicators
        if any(keyword in content for keyword in ["time", "when", "before", "after", "during"]):
            return ModalityType.TEMPORAL
        
        # Structured indicators
        if any(keyword in content for keyword in ["schema", "structure", "format", "template"]):
            return ModalityType.STRUCTURED
        
        # Default to text
        return ModalityType.TEXT
    
    def _find_cross_modal_similarities(
        self,
        memories1: List[MemoryEntry],
        memories2: List[MemoryEntry],
        modality1: ModalityType,
        modality2: ModalityType
    ) -> List[CrossModalLink]:
        """Find similarities between memories of different modalities."""
        links = []
        
        for mem1 in memories1:
            for mem2 in memories2:
                similarity = self._calculate_cross_modal_similarity(mem1, mem2)
                
                if similarity > 0.5:  # Threshold for creating link
                    link = CrossModalLink(
                        source_id=str(id(mem1)),  # Simplified ID
                        target_id=str(id(mem2)),
                        source_modality=modality1,
                        target_modality=modality2,
                        relation_type="semantic_similarity",
                        strength=similarity
                    )
                    links.append(link)
        
        return links
    
    def _calculate_cross_modal_similarity(self, mem1: MemoryEntry, mem2: MemoryEntry) -> float:
        """Calculate similarity between memories of different modalities."""
        # Simplified similarity calculation based on shared keywords
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class MemoryConsolidationEngine:
    """Handles memory consolidation and forgetting mechanisms."""
    
    def __init__(self, consolidation_interval: float = 3600.0):  # 1 hour
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = time.time()
        self.consolidation_rules: List[ConsolidationRule] = []
        self.clusters: Dict[str, MemoryCluster] = {}
        self._initialize_default_rules()
    
    def consolidate_memories(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Perform memory consolidation."""
        current_time = time.time()
        if current_time - self.last_consolidation < self.consolidation_interval:
            return {"status": "skipped", "reason": "too_soon"}
        
        consolidation_results = {
            "status": "completed",
            "clusters_created": 0,
            "memories_consolidated": 0,
            "memories_forgotten": 0,
            "strategies_applied": []
        }
        
        # Apply consolidation strategies
        for rule in self.consolidation_rules:
            result = self._apply_consolidation_rule(rule, memories)
            consolidation_results["strategies_applied"].append({
                "rule": rule.name,
                "result": result
            })
        
        # Create memory clusters
        new_clusters = self._create_memory_clusters(memories)
        self.clusters.update(new_clusters)
        consolidation_results["clusters_created"] = len(new_clusters)
        
        # Apply forgetting mechanisms
        forgotten_count = self._apply_forgetting_mechanisms(memories)
        consolidation_results["memories_forgotten"] = forgotten_count
        
        self.last_consolidation = current_time
        return consolidation_results
    
    def _initialize_default_rules(self) -> None:
        """Initialize default consolidation rules."""
        # Frequency-based consolidation
        self.consolidation_rules.append(ConsolidationRule(
            name="frequent_access",
            conditions=["access_count > 5"],
            actions=["increase_retention", "strengthen_links"],
            priority=0.8,
            strategy=ConsolidationStrategy.FREQUENCY_BASED
        ))
        
        # Recency-based consolidation
        self.consolidation_rules.append(ConsolidationRule(
            name="recent_memories",
            conditions=["age < 24_hours"],
            actions=["maintain_retention", "preserve_detail"],
            priority=0.7,
            strategy=ConsolidationStrategy.RECENCY_BASED
        ))
        
        # Importance-based consolidation
        self.consolidation_rules.append(ConsolidationRule(
            name="high_importance",
            conditions=["importance_score > 0.8"],
            actions=["permanent_retention", "strengthen_all_links"],
            priority=0.9,
            strategy=ConsolidationStrategy.IMPORTANCE_BASED
        ))
    
    def _apply_consolidation_rule(self, rule: ConsolidationRule, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Apply a consolidation rule to memories."""
        affected_memories = []
        
        for memory in memories:
            if self._memory_matches_conditions(memory, rule.conditions):
                affected_memories.append(memory)
                # Apply actions (simplified)
                for action in rule.actions:
                    self._apply_consolidation_action(memory, action)
        
        return {
            "affected_count": len(affected_memories),
            "strategy": rule.strategy.value,
            "actions": rule.actions
        }
    
    def _memory_matches_conditions(self, memory: MemoryEntry, conditions: List[str]) -> bool:
        """Check if memory matches consolidation conditions."""
        # Simplified condition checking
        for condition in conditions:
            if "access_count" in condition:
                access_count = memory.metadata.get("access_count", 0)
                if ">" in condition:
                    threshold = int(condition.split(">")[1].strip())
                    if access_count <= threshold:
                        return False
            elif "age" in condition:
                age_hours = (time.time() - memory.timestamp) / 3600
                if "< 24_hours" in condition and age_hours >= 24:
                    return False
            elif "importance_score" in condition:
                importance = memory.metadata.get("importance_score", 0.0)
                if ">" in condition:
                    threshold = float(condition.split(">")[1].strip())
                    if importance <= threshold:
                        return False
        
        return True
    
    def _apply_consolidation_action(self, memory: MemoryEntry, action: str) -> None:
        """Apply a consolidation action to a memory."""
        if action == "increase_retention":
            memory.metadata["retention_boost"] = memory.metadata.get("retention_boost", 0) + 0.1
        elif action == "strengthen_links":
            memory.metadata["link_strength"] = memory.metadata.get("link_strength", 1.0) * 1.2
        elif action == "permanent_retention":
            memory.metadata["permanent"] = True
        elif action == "maintain_retention":
            memory.metadata["maintain_until"] = time.time() + 7 * 24 * 3600  # 7 days
    
    def _create_memory_clusters(self, memories: List[MemoryEntry]) -> Dict[str, MemoryCluster]:
        """Create clusters of related memories."""
        clusters = {}
        
        # Simple clustering based on content similarity
        unclustered = memories.copy()
        cluster_id = 0
        
        while unclustered:
            seed_memory = unclustered.pop(0)
            cluster_memories = [str(id(seed_memory))]
            
            # Find similar memories
            similar_memories = []
            for memory in unclustered.copy():
                if self._memories_are_similar(seed_memory, memory):
                    similar_memories.append(memory)
                    cluster_memories.append(str(id(memory)))
                    unclustered.remove(memory)
            
            # Create cluster if it has multiple memories
            if len(cluster_memories) > 1:
                cluster = MemoryCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    memories=cluster_memories,
                    coherence_score=self._calculate_cluster_coherence(seed_memory, similar_memories),
                    metadata={"size": len(cluster_memories), "seed_content": seed_memory.content[:100]}
                )
                clusters[cluster.cluster_id] = cluster
                cluster_id += 1
        
        return clusters
    
    def _memories_are_similar(self, mem1: MemoryEntry, mem2: MemoryEntry, threshold: float = 0.3) -> bool:
        """Check if two memories are similar enough to cluster."""
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        return similarity >= threshold
    
    def _calculate_cluster_coherence(self, seed: MemoryEntry, cluster_memories: List[MemoryEntry]) -> float:
        """Calculate coherence score for a memory cluster."""
        if not cluster_memories:
            return 1.0
        
        total_similarity = 0.0
        count = 0
        
        for memory in cluster_memories:
            similarity = self._calculate_cross_modal_similarity(seed, memory)
            total_similarity += similarity
            count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def _calculate_cross_modal_similarity(self, mem1: MemoryEntry, mem2: MemoryEntry) -> float:
        """Calculate similarity between memories."""
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_forgetting_mechanisms(self, memories: List[MemoryEntry]) -> int:
        """Apply forgetting mechanisms to remove or weaken old memories."""
        forgotten_count = 0
        current_time = time.time()
        
        for memory in memories:
            # Check if memory should be forgotten
            age_hours = (current_time - memory.timestamp) / 3600
            
            # Don't forget permanent memories
            if memory.metadata.get("permanent", False):
                continue
            
            # Forget very old, unimportant memories
            if (age_hours > 168 and  # Older than 1 week
                memory.metadata.get("access_count", 0) < 2 and  # Rarely accessed
                memory.metadata.get("importance_score", 0.0) < 0.3):  # Low importance
                
                memory.metadata["forgotten"] = True
                forgotten_count += 1
            
            # Weaken old memories
            elif age_hours > 72:  # Older than 3 days
                decay_factor = max(0.1, 1.0 - (age_hours - 72) / (168 - 72))
                memory.metadata["retention_strength"] = memory.metadata.get("retention_strength", 1.0) * decay_factor
        
        return forgotten_count


class EnhancedEpisodicMemory(MemoryLayer):
    """Enhanced episodic memory with temporal reasoning and cross-modal linking."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_path = Path(config.get("storage_path", "sessions/enhanced_episodic.json"))
        self.max_episodes = config.get("max_episodes", 2000)
        
        # Enhanced components
        self.knowledge_organizer = HierarchicalKnowledgeOrganizer(
            max_depth=config.get("hierarchy_depth", 5)
        )
        self.cross_modal_linker = CrossModalMemoryLinker()
        self.consolidation_engine = MemoryConsolidationEngine(
            consolidation_interval=config.get("consolidation_interval", 3600.0)
        )
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._episodes: List[Dict[str, Any]] = self._load_episodes()
        self._initialize_enhanced_features()
    
    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.EPISODIC
    
    def store(self, entry: MemoryEntry) -> bool:
        """Store enhanced episodic memory entry."""
        # Create episode
        episode = {
            "id": f"ep_{int(time.time() * 1000)}_{len(self._episodes)}",
            "content": entry.content,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp,
            "agent_name": entry.agent_name,
            "tags": entry.tags or [],
            "modality": self._infer_modality(entry).value,
            "access_count": 0,
            "importance_score": self._calculate_importance(entry),
        }
        
        self._episodes.append(episode)
        
        # Enforce episode limit
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]
        
        # Trigger consolidation if needed
        memories = [self._episode_to_memory_entry(ep) for ep in self._episodes]
        consolidation_result = self.consolidation_engine.consolidate_memories(memories)
        
        # Update cross-modal links
        if len(self._episodes) % 10 == 0:  # Every 10 episodes
            self._update_cross_modal_links()
        
        # Persist to storage
        self._save_episodes()
        return True
    
    def clear(self) -> bool:
        """Clear all episodes from memory."""
        try:
            self._episodes.clear()
            self.knowledge_organizer = HierarchicalKnowledgeOrganizer(
                max_depth=self.config.get("hierarchy_depth", 5)
            )
            self.cross_modal_linker = CrossModalMemoryLinker()
            self.consolidation_engine = MemoryConsolidationEngine(
                consolidation_interval=self.config.get("consolidation_interval", 3600.0)
            )
            self._save_episodes()
            logger.info(f"Cleared all episodes from enhanced memory")
            return True
        except Exception as e:
            logger.error(f"Failed to clear enhanced memory: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        time_range: Optional[Tuple[float, float]] = None,
        modality_filter: Optional[ModalityType] = None,
        use_temporal_reasoning: bool = True
    ) -> MemoryRetrieval:
        """Enhanced retrieval with temporal reasoning and cross-modal linking."""
        start_time = time.time()
        
        # Filter episodes
        filtered_episodes = self._filter_episodes(query, time_range, modality_filter)
        
        # Apply temporal reasoning if requested
        if use_temporal_reasoning and len(filtered_episodes) > 1:
            filtered_episodes = self._apply_temporal_reasoning(filtered_episodes, query)
        
        # Find cross-modal links
        linked_episodes = self._find_cross_modal_episodes(filtered_episodes)
        
        # Combine and rank results
        all_episodes = filtered_episodes.copy()
        
        # Add linked episodes that aren't already in filtered_episodes
        filtered_ids = {ep["id"] for ep in filtered_episodes}
        for linked_ep in linked_episodes:
            if linked_ep["id"] not in filtered_ids:
                all_episodes.append(linked_ep)
        
        ranked_episodes = self._rank_episodes(all_episodes, query)
        
        # Convert to memory entries
        entries = []
        for episode in ranked_episodes[:limit]:
            memory_entry = self._episode_to_memory_entry(episode)
            entries.append(memory_entry)
            # Update access count
            episode["access_count"] = episode.get("access_count", 0) + 1
        
        # Update storage with access counts
        self._save_episodes()
        
        retrieval_time = time.time() - start_time
        total_tokens = sum(len(entry.content.split()) for entry in entries)
        
        return MemoryRetrieval(
            entries=entries,
            total_tokens=total_tokens,
            retrieval_time=retrieval_time,
            source_layers=[MemoryType.EPISODIC]
        )
    
    def get_memory_hierarchy(self) -> Dict[str, Any]:
        """Get hierarchical organization of memories."""
        memories = [self._episode_to_memory_entry(ep) for ep in self._episodes]
        return self.knowledge_organizer.organize_memories(memories)
    
    def get_cross_modal_links(self, memory_id: str) -> List[CrossModalLink]:
        """Get cross-modal links for a specific memory."""
        return self.cross_modal_linker.find_links(memory_id)
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status."""
        return {
            "last_consolidation": self.consolidation_engine.last_consolidation,
            "clusters_count": len(self.consolidation_engine.clusters),
            "consolidation_rules": len(self.consolidation_engine.consolidation_rules),
            "next_consolidation": self.consolidation_engine.last_consolidation + 
                                self.consolidation_engine.consolidation_interval
        }
    
    def force_consolidate(self) -> Dict[str, Any]:
        """Force memory consolidation to run immediately, bypassing time interval check."""
        memories = [self._episode_to_memory_entry(ep) for ep in self._episodes]
        
        # Temporarily store the last consolidation time
        original_last_consolidation = self.consolidation_engine.last_consolidation
        
        # Set last consolidation to a time that ensures consolidation will run
        self.consolidation_engine.last_consolidation = 0
        
        # Run consolidation
        result = self.consolidation_engine.consolidate_memories(memories)
        
        # If consolidation was skipped for any reason other than time, restore original time
        if result.get("status") == "skipped" and result.get("reason") != "too_soon":
            self.consolidation_engine.last_consolidation = original_last_consolidation
        
        return result
    
    def _initialize_enhanced_features(self) -> None:
        """Initialize enhanced memory features."""
        if self._episodes:
            memories = [self._episode_to_memory_entry(ep) for ep in self._episodes]
            
            # Initialize knowledge organization
            self.knowledge_organizer.organize_memories(memories)
            
            # Initialize cross-modal links
            self.cross_modal_linker.create_automatic_links(memories)
    
    def _infer_modality(self, entry: MemoryEntry) -> ModalityType:
        """Infer modality type from memory entry."""
        return self.cross_modal_linker._infer_modality(entry)
    
    def _calculate_importance(self, entry: MemoryEntry) -> float:
        """Calculate importance score for memory entry."""
        importance = 0.5  # Base importance
        
        # Boost for agent-generated content
        if entry.agent_name:
            importance += 0.1
        
        # Boost for tagged content
        if entry.tags:
            importance += 0.05 * len(entry.tags)
        
        # Boost for metadata richness
        if entry.metadata:
            importance += 0.02 * len(entry.metadata)
        
        # Boost for content length (up to a point)
        content_length = len(entry.content)
        if content_length > 100:
            importance += min(0.2, content_length / 1000)
        
        return min(1.0, importance)
    
    def _episode_to_memory_entry(self, episode: Dict[str, Any]) -> MemoryEntry:
        """Convert episode to memory entry."""
        return MemoryEntry(
            content=episode["content"],
            metadata=episode["metadata"],
            timestamp=episode["timestamp"],
            agent_name=episode.get("agent_name"),
            tags=episode.get("tags", [])
        )
    
    def _filter_episodes(
        self,
        query: str,
        time_range: Optional[Tuple[float, float]],
        modality_filter: Optional[ModalityType]
    ) -> List[Dict[str, Any]]:
        """Filter episodes based on query and constraints."""
        filtered = []
        query_words = set(query.lower().split())
        
        for episode in self._episodes:
            # Skip forgotten memories
            if episode.get("forgotten", False):
                continue
            
            # Time filter
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= episode["timestamp"] <= end_time):
                    continue
            
            # Modality filter
            if modality_filter and episode.get("modality") != modality_filter.value:
                continue
            
            # Content relevance
            episode_words = set(episode["content"].lower().split())
            if query_words.intersection(episode_words):
                filtered.append(episode)
        
        return filtered
    
    def _apply_temporal_reasoning(self, episodes: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply temporal reasoning to episode selection."""
        # Sort by timestamp
        episodes.sort(key=lambda x: x["timestamp"])
        
        # Look for temporal patterns
        if any(word in query.lower() for word in ["before", "after", "during", "sequence"]):
            # Return episodes in temporal order
            return episodes
        else:
            # Return episodes weighted by recency and relevance
            current_time = time.time()
            scored_episodes = []
            
            for episode in episodes:
                recency_score = 1.0 / (1.0 + (current_time - episode["timestamp"]) / 86400)  # Decay over days
                importance_score = episode.get("importance_score", 0.5)
                access_score = min(1.0, episode.get("access_count", 0) / 10.0)
                
                total_score = (recency_score + importance_score + access_score) / 3.0
                scored_episodes.append((episode, total_score))
            
            # Sort by score and return episodes
            scored_episodes.sort(key=lambda x: x[1], reverse=True)
            return [ep for ep, score in scored_episodes]
    
    def _find_cross_modal_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find episodes linked through cross-modal connections."""
        linked_episodes = []
        
        for episode in episodes:
            episode_id = episode["id"]
            links = self.cross_modal_linker.find_links(episode_id)
            
            for link in links:
                # Find linked episode
                target_id = link.target_id if link.source_id == episode_id else link.source_id
                
                for ep in self._episodes:
                    if ep["id"] == target_id and ep not in episodes and ep not in linked_episodes:
                        linked_episodes.append(ep)
                        break
        
        return linked_episodes
    
    def _rank_episodes(self, episodes: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank episodes by relevance to query."""
        query_words = set(query.lower().split())
        scored_episodes = []
        
        for episode in episodes:
            # Content relevance score
            episode_words = set(episode["content"].lower().split())
            content_overlap = len(query_words.intersection(episode_words))
            content_score = content_overlap / len(query_words) if query_words else 0.0
            
            # Importance score
            importance_score = episode.get("importance_score", 0.5)
            
            # Access frequency score
            access_score = min(1.0, episode.get("access_count", 0) / 5.0)
            
            # Retention strength
            retention_score = episode.get("retention_strength", 1.0)
            
            # Combined score
            total_score = (content_score * 0.4 + importance_score * 0.3 + 
                          access_score * 0.2 + retention_score * 0.1)
            
            scored_episodes.append((episode, total_score))
        
        # Sort by score
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, score in scored_episodes]
    
    def _update_cross_modal_links(self) -> None:
        """Update cross-modal links for recent episodes."""
        recent_episodes = [ep for ep in self._episodes[-20:]]  # Last 20 episodes
        memories = [self._episode_to_memory_entry(ep) for ep in recent_episodes]
        
        new_links = self.cross_modal_linker.create_automatic_links(memories)
        logger.info(f"Created {len(new_links)} new cross-modal links")
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episodes from storage."""
        if not self.storage_path.exists():
            return []
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return data.get("episodes", [])
        except Exception as e:
            logger.warning(f"Failed to load episodes: {e}")
            return []
    
    def _save_episodes(self) -> None:
        """Save episodes to storage."""
        try:
            data = {
                "episodes": self._episodes,
                "metadata": {
                    "last_updated": time.time(),
                    "episode_count": len(self._episodes),
                    "consolidation_status": self.get_consolidation_status()
                }
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save episodes: {e}")
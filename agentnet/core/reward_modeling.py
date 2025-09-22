"""
Reward Modeling and Offline Evaluation Loops Module

Implements reward modeling with feedback collection and offline evaluation
for continuous agent improvement and reinforcement learning integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    HUMAN_RATING = "human_rating"
    AUTOMATED_SCORE = "automated_score"
    PERFORMANCE_METRIC = "performance_metric"
    QUALITY_ASSESSMENT = "quality_assessment"
    SAFETY_VIOLATION = "safety_violation"
    USER_SATISFACTION = "user_satisfaction"


class RewardSignal(str, Enum):
    """Types of reward signals."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SPARSE = "sparse"  # Delayed/infrequent rewards


@dataclass
class FeedbackEntry:
    """A single feedback entry for reward modeling."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context
    session_id: str = ""
    agent_id: str = ""
    action_taken: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Feedback
    feedback_type: FeedbackType = FeedbackType.HUMAN_RATING
    reward_signal: RewardSignal = RewardSignal.NEUTRAL
    score: float = 0.0  # Normalized score [-1, 1]
    raw_score: Optional[float] = None  # Original score before normalization
    feedback_source: str = ""
    feedback_text: Optional[str] = None
    
    # Metadata
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationBatch:
    """A batch of evaluations for offline processing."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    feedback_entries: List[FeedbackEntry] = field(default_factory=list)
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    processed: bool = False


class RewardModel:
    """
    Reward modeling system for agent feedback and improvement.
    
    Features:
    - Collects feedback from multiple sources
    - Normalizes and aggregates rewards
    - Provides scoring for actions and outputs
    - Supports offline evaluation loops
    """
    
    def __init__(
        self,
        feedback_window: timedelta = timedelta(hours=24),
        min_feedback_count: int = 5,
        reward_decay: float = 0.95,
        enable_online_learning: bool = False
    ):
        self.feedback_window = feedback_window
        self.min_feedback_count = min_feedback_count
        self.reward_decay = reward_decay
        self.enable_online_learning = enable_online_learning
        
        self.feedback_store: List[FeedbackEntry] = []
        self.reward_history: Dict[str, List[float]] = {}  # agent_id -> scores
        self.evaluation_batches: List[EvaluationBatch] = []
        
        # Learned reward functions
        self.reward_functions: Dict[str, Callable[[Dict[str, Any]], float]] = {}
        
        # Callbacks
        self.on_feedback_received: Optional[Callable[[FeedbackEntry], None]] = None
        self.on_batch_processed: Optional[Callable[[EvaluationBatch], None]] = None
        
        logger.info("RewardModel initialized")
    
    def add_feedback(
        self,
        session_id: str,
        agent_id: str,
        action_taken: str,
        feedback_type: FeedbackType,
        score: float,
        feedback_source: str = "",
        context: Optional[Dict[str, Any]] = None,
        feedback_text: Optional[str] = None,
        confidence: float = 1.0
    ) -> str:
        """Add feedback entry to the reward model."""
        
        # Normalize score to [-1, 1] range
        normalized_score = self._normalize_score(score, feedback_type)
        
        # Determine reward signal
        reward_signal = self._determine_reward_signal(normalized_score)
        
        feedback = FeedbackEntry(
            session_id=session_id,
            agent_id=agent_id,
            action_taken=action_taken,
            feedback_type=feedback_type,
            reward_signal=reward_signal,
            score=normalized_score,
            raw_score=score,
            feedback_source=feedback_source,
            feedback_text=feedback_text,
            context=context or {},
            confidence=confidence
        )
        
        self.feedback_store.append(feedback)
        
        # Update reward history
        if agent_id not in self.reward_history:
            self.reward_history[agent_id] = []
        self.reward_history[agent_id].append(normalized_score)
        
        # Trigger callback
        if self.on_feedback_received:
            self.on_feedback_received(feedback)
        
        # Online learning update
        if self.enable_online_learning:
            self._update_reward_function(feedback)
        
        logger.debug(f"Added feedback for agent {agent_id}: {normalized_score}")
        
        return feedback.id
    
    def get_agent_reward_score(self, agent_id: str, window: Optional[timedelta] = None) -> Optional[float]:
        """Get current reward score for an agent."""
        
        window = window or self.feedback_window
        cutoff_time = datetime.now() - window
        
        # Get recent feedback for agent
        recent_feedback = [
            feedback for feedback in self.feedback_store
            if feedback.agent_id == agent_id and feedback.timestamp >= cutoff_time
        ]
        
        if len(recent_feedback) < self.min_feedback_count:
            return None
        
        # Calculate weighted average with time decay
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for feedback in recent_feedback:
            # Time decay factor
            time_diff = (datetime.now() - feedback.timestamp).total_seconds()
            time_decay = self.reward_decay ** (time_diff / 3600)  # Decay per hour
            
            weight = feedback.confidence * time_decay
            total_weighted_score += feedback.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def predict_reward(self, agent_id: str, action_context: Dict[str, Any]) -> float:
        """Predict reward for a potential action."""
        
        # Use learned reward function if available
        if agent_id in self.reward_functions:
            return self.reward_functions[agent_id](action_context)
        
        # Fallback to historical average
        if agent_id in self.reward_history and self.reward_history[agent_id]:
            return statistics.mean(self.reward_history[agent_id][-10:])  # Last 10 scores
        
        return 0.0  # Neutral prediction
    
    def create_evaluation_batch(self, days_back: int = 1) -> EvaluationBatch:
        """Create a batch of recent feedback for offline evaluation."""
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        # Collect recent feedback
        recent_feedback = [
            feedback for feedback in self.feedback_store
            if feedback.timestamp >= cutoff_time
        ]
        
        batch = EvaluationBatch(feedback_entries=recent_feedback)
        
        # Calculate batch metrics
        if recent_feedback:
            scores = [feedback.score for feedback in recent_feedback]
            batch.evaluation_metrics = {
                "total_feedback": len(recent_feedback),
                "average_score": statistics.mean(scores),
                "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "positive_feedback_ratio": len([s for s in scores if s > 0]) / len(scores),
                "agent_coverage": len(set(feedback.agent_id for feedback in recent_feedback))
            }
        
        self.evaluation_batches.append(batch)
        
        logger.info(f"Created evaluation batch {batch.id} with {len(recent_feedback)} entries")
        
        return batch
    
    async def process_evaluation_batch(self, batch_id: str) -> Dict[str, Any]:
        """Process an evaluation batch for offline learning."""
        
        batch = next((b for b in self.evaluation_batches if b.id == batch_id), None)
        
        if not batch or batch.processed:
            return {}
        
        logger.info(f"Processing evaluation batch {batch_id}")
        
        # Group feedback by agent
        agent_feedback = {}
        for feedback in batch.feedback_entries:
            if feedback.agent_id not in agent_feedback:
                agent_feedback[feedback.agent_id] = []
            agent_feedback[feedback.agent_id].append(feedback)
        
        # Process each agent's feedback
        results = {}
        for agent_id, feedback_list in agent_feedback.items():
            agent_results = await self._process_agent_feedback(agent_id, feedback_list)
            results[agent_id] = agent_results
        
        # Update learned reward functions
        await self._update_reward_functions(agent_feedback)
        
        # Mark batch as processed
        batch.processed = True
        
        # Trigger callback
        if self.on_batch_processed:
            self.on_batch_processed(batch)
        
        logger.info(f"Completed processing batch {batch_id}")
        
        return results
    
    def get_learning_insights(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get insights about reward learning and agent performance."""
        
        if agent_id:
            # Single agent insights
            if agent_id not in self.reward_history:
                return {"error": "No data for agent"}
            
            scores = self.reward_history[agent_id]
            recent_scores = scores[-20:] if len(scores) > 20 else scores
            
            return {
                "agent_id": agent_id,
                "total_feedback": len(scores),
                "current_average": statistics.mean(recent_scores) if recent_scores else 0.0,
                "improvement_trend": self._calculate_trend(scores),
                "best_score": max(scores) if scores else 0.0,
                "worst_score": min(scores) if scores else 0.0,
                "consistency": 1.0 - (statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0)
            }
        else:
            # Global insights
            all_agents = list(self.reward_history.keys())
            total_feedback = sum(len(scores) for scores in self.reward_history.values())
            
            # Calculate global metrics
            all_recent_scores = []
            for scores in self.reward_history.values():
                recent = scores[-10:] if len(scores) > 10 else scores
                all_recent_scores.extend(recent)
            
            return {
                "total_agents": len(all_agents),
                "total_feedback": total_feedback,
                "global_average": statistics.mean(all_recent_scores) if all_recent_scores else 0.0,
                "top_performing_agents": self._get_top_agents(5),
                "feedback_distribution": self._get_feedback_distribution(),
                "evaluation_batches_processed": len([b for b in self.evaluation_batches if b.processed])
            }
    
    def export_feedback_data(self, filepath: str, days_back: int = 30) -> None:
        """Export feedback data for external analysis."""
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "days_back": days_back,
            "feedback_entries": [
                {
                    "id": feedback.id,
                    "timestamp": feedback.timestamp.isoformat(),
                    "session_id": feedback.session_id,
                    "agent_id": feedback.agent_id,
                    "action_taken": feedback.action_taken,
                    "feedback_type": feedback.feedback_type,
                    "reward_signal": feedback.reward_signal,
                    "score": feedback.score,
                    "raw_score": feedback.raw_score,
                    "feedback_source": feedback.feedback_source,
                    "confidence": feedback.confidence,
                    "context": feedback.context,
                    "metadata": feedback.metadata
                }
                for feedback in self.feedback_store
                if feedback.timestamp >= cutoff_time
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data['feedback_entries'])} feedback entries to {filepath}")
    
    def _normalize_score(self, score: float, feedback_type: FeedbackType) -> float:
        """Normalize score to [-1, 1] range based on feedback type."""
        
        # Type-specific normalization
        if feedback_type == FeedbackType.HUMAN_RATING:
            # Assume rating is 1-5, normalize to [-1, 1]
            return (score - 3.0) / 2.0
        elif feedback_type == FeedbackType.USER_SATISFACTION:
            # Assume satisfaction is 0-10, normalize to [-1, 1]
            return (score - 5.0) / 5.0
        elif feedback_type == FeedbackType.SAFETY_VIOLATION:
            # Safety violations are negative
            return -abs(score)
        else:
            # Default: assume score is already in reasonable range
            return max(-1.0, min(1.0, score))
    
    def _determine_reward_signal(self, normalized_score: float) -> RewardSignal:
        """Determine reward signal type from normalized score."""
        
        if normalized_score > 0.1:
            return RewardSignal.POSITIVE
        elif normalized_score < -0.1:
            return RewardSignal.NEGATIVE
        else:
            return RewardSignal.NEUTRAL
    
    def _update_reward_function(self, feedback: FeedbackEntry) -> None:
        """Update learned reward function with new feedback (simplified)."""
        
        # This is a placeholder for more sophisticated learning algorithms
        # In practice, this would use ML techniques like neural networks, 
        # gradient boosting, or other regression/classification methods
        
        agent_id = feedback.agent_id
        
        if agent_id not in self.reward_functions:
            # Create simple reward function based on context patterns
            self.reward_functions[agent_id] = self._create_simple_reward_function(agent_id)
    
    def _create_simple_reward_function(self, agent_id: str) -> Callable[[Dict[str, Any]], float]:
        """Create a simple reward function based on historical data."""
        
        def reward_function(context: Dict[str, Any]) -> float:
            # Simple heuristic-based reward function
            # In practice, this would be a trained ML model
            
            base_score = 0.0
            
            # Context-based adjustments
            complexity = context.get("complexity", 0.5)
            if complexity > 0.7:
                base_score -= 0.1  # Penalize overly complex actions
            
            confidence = context.get("confidence", 0.5)
            base_score += (confidence - 0.5) * 0.2
            
            # Agent-specific historical performance
            if agent_id in self.reward_history:
                recent_avg = statistics.mean(self.reward_history[agent_id][-5:])
                base_score += recent_avg * 0.3
            
            return max(-1.0, min(1.0, base_score))
        
        return reward_function
    
    async def _process_agent_feedback(self, agent_id: str, feedback_list: List[FeedbackEntry]) -> Dict[str, Any]:
        """Process feedback for a specific agent."""
        
        scores = [feedback.score for feedback in feedback_list]
        
        # Calculate metrics
        avg_score = statistics.mean(scores) if scores else 0.0
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        
        # Identify patterns in feedback
        positive_actions = [
            feedback.action_taken for feedback in feedback_list
            if feedback.reward_signal == RewardSignal.POSITIVE
        ]
        
        negative_actions = [
            feedback.action_taken for feedback in feedback_list
            if feedback.reward_signal == RewardSignal.NEGATIVE
        ]
        
        return {
            "agent_id": agent_id,
            "feedback_count": len(feedback_list),
            "average_score": avg_score,
            "score_variance": score_variance,
            "improvement_suggestions": self._generate_improvement_suggestions(feedback_list),
            "top_positive_actions": list(set(positive_actions))[:3],
            "top_negative_actions": list(set(negative_actions))[:3]
        }
    
    async def _update_reward_functions(self, agent_feedback: Dict[str, List[FeedbackEntry]]) -> None:
        """Update reward functions based on batch processing."""
        
        for agent_id, feedback_list in agent_feedback.items():
            if len(feedback_list) >= self.min_feedback_count:
                # Update or create reward function
                self.reward_functions[agent_id] = self._create_simple_reward_function(agent_id)
                logger.debug(f"Updated reward function for agent {agent_id}")
    
    def _generate_improvement_suggestions(self, feedback_list: List[FeedbackEntry]) -> List[str]:
        """Generate improvement suggestions based on feedback patterns."""
        
        suggestions = []
        
        # Analyze feedback patterns
        negative_feedback = [f for f in feedback_list if f.score < -0.1]
        
        if len(negative_feedback) > len(feedback_list) * 0.3:
            suggestions.append("High negative feedback rate - review action selection strategy")
        
        # Check for consistency issues
        scores = [f.score for f in feedback_list]
        if len(scores) > 2 and statistics.stdev(scores) > 0.5:
            suggestions.append("Inconsistent performance - consider more stable decision making")
        
        # Safety-related suggestions
        safety_violations = [f for f in feedback_list if f.feedback_type == FeedbackType.SAFETY_VIOLATION]
        if safety_violations:
            suggestions.append("Safety violations detected - review policy compliance")
        
        return suggestions
    
    def _calculate_trend(self, scores: List[float], window: int = 10) -> float:
        """Calculate improvement trend from recent scores."""
        
        if len(scores) < window:
            return 0.0
        
        recent = scores[-window:]
        older = scores[-(window*2):-window] if len(scores) >= window*2 else scores[:-window]
        
        if not older:
            return 0.0
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        return recent_avg - older_avg
    
    def _get_top_agents(self, count: int) -> List[Dict[str, Any]]:
        """Get top performing agents by recent average score."""
        
        agent_scores = []
        
        for agent_id, scores in self.reward_history.items():
            if scores:
                recent_scores = scores[-10:] if len(scores) > 10 else scores
                avg_score = statistics.mean(recent_scores)
                agent_scores.append({"agent_id": agent_id, "average_score": avg_score})
        
        # Sort by average score
        agent_scores.sort(key=lambda x: x["average_score"], reverse=True)
        
        return agent_scores[:count]
    
    def _get_feedback_distribution(self) -> Dict[str, int]:
        """Get distribution of feedback types."""
        
        distribution = {}
        
        for feedback in self.feedback_store:
            feedback_type = feedback.feedback_type
            if feedback_type not in distribution:
                distribution[feedback_type] = 0
            distribution[feedback_type] += 1
        
        return distribution
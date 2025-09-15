"""
Analytics utilities for AgentNet experiments.
Provides functions for computing metrics, diversity indices, and analytics.
"""

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def compute_lexical_diversity(text: str) -> float:
    """Compute lexical diversity as unique tokens / total tokens."""
    if not text:
        return 0.0
    
    tokens = re.findall(r'\w+', text.lower())
    if not tokens:
        return 0.0
    
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    return unique_tokens / total_tokens


def compute_repetition_score(texts: List[str], window_size: int = 3) -> float:
    """Compute repetition score across a list of texts using sliding window."""
    if len(texts) < window_size:
        return 0.0
    
    similarities = []
    for i in range(len(texts) - window_size + 1):
        window = texts[i:i + window_size]
        # Simple Jaccard similarity for each pair in window
        total_sim = 0
        pairs = 0
        for j in range(len(window)):
            for k in range(j + 1, len(window)):
                sim = jaccard_similarity(window[j], window[k])
                total_sim += sim
                pairs += 1
        if pairs > 0:
            similarities.append(total_sim / pairs)
    
    return sum(similarities) / len(similarities) if similarities else 0.0


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    tokens1 = set(re.findall(r'\w+', text1.lower()))
    tokens2 = set(re.findall(r'\w+', text2.lower()))
    
    if not tokens1 and not tokens2:
        return 1.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def extract_session_metrics(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardized metrics from a session record."""
    metrics = {
        "runtime_seconds": 0.0,
        "confidence_score": 0.0,
        "token_count": 0,
        "violation_count": 0,
        "severe_violations": 0,
        "convergence_rate": 0.0,
        "lexical_diversity": 0.0,
        "rounds_executed": 0,
        "style_insights_count": 0
    }
    
    # Extract basic metrics
    if "runtime_seconds" in session_data:
        metrics["runtime_seconds"] = session_data["runtime_seconds"]
    
    if "rounds_executed" in session_data:
        metrics["rounds_executed"] = session_data["rounds_executed"]
    
    if "converged" in session_data:
        metrics["convergence_rate"] = 1.0 if session_data["converged"] else 0.0
    
    # Extract metrics from transcript
    transcript = session_data.get("transcript", [])
    if transcript:
        all_content = []
        confidences = []
        total_tokens = 0
        style_insights_total = 0
        
        for turn in transcript:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                if content:
                    all_content.append(content)
                    total_tokens += len(content.split())
                
                if "confidence" in turn:
                    confidences.append(turn["confidence"])
                
                if "style_insights" in turn:
                    style_insights_total += len(turn["style_insights"])
        
        # Compute derived metrics
        if confidences:
            metrics["confidence_score"] = sum(confidences) / len(confidences)
        
        metrics["token_count"] = total_tokens
        metrics["style_insights_count"] = style_insights_total
        
        if all_content:
            combined_text = " ".join(all_content)
            metrics["lexical_diversity"] = compute_lexical_diversity(combined_text)
    
    # Extract violation metrics
    violations = session_data.get("violations", [])
    if violations:
        metrics["violation_count"] = len(violations)
        severe_count = sum(1 for v in violations if v.get("severity") == "severe")
        metrics["severe_violations"] = severe_count
    
    return metrics


def write_metrics_jsonl(metrics_data: Dict[str, Any], output_path: Path) -> None:
    """Write metrics data to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp if not present
    if "timestamp" not in metrics_data:
        metrics_data["timestamp"] = datetime.utcnow().isoformat()
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_data) + "\n")


def load_metrics_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Load metrics data from JSONL file."""
    if not input_path.exists():
        return []
    
    metrics = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return metrics


def compute_diversity_index(sessions: List[Dict[str, Any]]) -> float:
    """Compute Shannon diversity index across multiple sessions."""
    if not sessions:
        return 0.0
    
    # Collect all tokens from all sessions
    all_tokens = []
    for session in sessions:
        transcript = session.get("transcript", [])
        for turn in transcript:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                tokens = re.findall(r'\w+', content.lower())
                all_tokens.extend(tokens)
    
    if not all_tokens:
        return 0.0
    
    # Compute Shannon entropy
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    entropy = 0.0
    for count in token_counts.values():
        p = count / total_tokens
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by maximum possible entropy
    unique_tokens = len(token_counts)
    max_entropy = math.log2(unique_tokens) if unique_tokens > 1 else 1.0
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def scan_session_directory(session_dir: Path) -> List[Dict[str, Any]]:
    """Scan a directory for session JSON files and extract metrics."""
    if not session_dir.exists():
        return []
    
    sessions = []
    for json_file in session_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                sessions.append(session_data)
        except (json.JSONDecodeError, IOError):
            continue
    
    return sessions
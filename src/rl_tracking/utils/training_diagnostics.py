"""Diagnostic utilities for training analysis."""
import numpy as np
import torch
from typing import Dict, List


def analyze_reward_distribution(rewards: List[float]) -> Dict:
    """Analyze the distribution of rewards.
    
    Args:
        rewards: List of reward values
        
    Returns:
        Dictionary with statistics
    """
    rewards_array = np.array(rewards)
    return {
        'mean': float(np.mean(rewards_array)),
        'std': float(np.std(rewards_array)),
        'min': float(np.min(rewards_array)),
        'max': float(np.max(rewards_array)),
        'median': float(np.median(rewards_array)),
        'percentile_25': float(np.percentile(rewards_array, 25)),
        'percentile_75': float(np.percentile(rewards_array, 75)),
        'zero_count': int(np.sum(rewards_array == 0)),
        'zero_percentage': float(100 * np.sum(rewards_array == 0) / len(rewards_array)),
    }


def analyze_q_values(q_values: torch.Tensor, actions: torch.Tensor) -> Dict:
    """Analyze Q-value distribution.
    
    Args:
        q_values: Tensor of Q-values [batch_size, num_actions]
        actions: Tensor of selected actions [batch_size]
        
    Returns:
        Dictionary with statistics
    """
    with torch.no_grad():
        selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        max_q = q_values.max(1)[0]
        min_q = q_values.min(1)[0]
        
        return {
            'selected_q_mean': float(selected_q.mean().item()),
            'selected_q_std': float(selected_q.std().item()),
            'selected_q_min': float(selected_q.min().item()),
            'selected_q_max': float(selected_q.max().item()),
            'max_q_mean': float(max_q.mean().item()),
            'max_q_std': float(max_q.std().item()),
            'min_q_mean': float(min_q.mean().item()),
            'q_value_range': float((max_q - min_q).mean().item()),
        }


def check_gradient_flow(model) -> Dict:
    """Check gradient flow through the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with gradient statistics
    """
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = {
                'mean': float(param.grad.mean().item()),
                'std': float(param.grad.std().item()),
                'max': float(param.grad.max().item()),
                'min': float(param.grad.min().item()),
            }
    return grad_stats


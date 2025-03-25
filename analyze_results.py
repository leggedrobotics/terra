#!/usr/bin/env python3
"""
Analyze and visualize results for the AutonomousExcavatorGame tested with different models.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("autonomous-excavator.analysis")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze AutonomousExcavatorGame results')
    parser.add_argument('--input_dir', default='./experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='./analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to include in analysis (default: all models)')
    return parser.parse_args()

def find_experiment_dirs(input_dir, models=None):
    """Find experiment directories for the AutonomousExcavatorGame."""
    experiment_dirs = []
    game = "AutonomousExcavatorGame"  # Hardcoded game name
    
    # List all directories in the input directory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Parse the directory name to extract the model
        if not item.startswith(game + "_"):
            continue
        
        model = item[len(game) + 1:]  # Extract model name after "AutonomousExcavatorGame_"
        
        # Filter by model if specified
        if models and model not in models:
            continue
            
        experiment_dirs.append((item_path, model))
    
    return experiment_dirs

def load_results(experiment_dir):
    """Load results from an experiment directory."""
    results = {}
    
    # Try to load actions and rewards
    csv_path = os.path.join(experiment_dir, 'actions_rewards.csv')
    if os.path.exists(csv_path):
        actions = []
        rewards = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        actions.append(int(row[0]))
                        rewards.append(float(row[1]))
            results['actions'] = actions
            results['cumulative_rewards'] = rewards
        except Exception as e:
            logger.error(f"Error loading CSV from {csv_path}: {str(e)}")
    
    return results

def analyze_results(experiment_dirs):
    """Analyze results from all experiment directories."""
    analysis = defaultdict(dict)
    
    for exp_dir, model in experiment_dirs:
        logger.info(f"Analyzing results for model {model}")
        results = load_results(exp_dir)
        
        if not results:
            logger.warning(f"No results found in {exp_dir}")
            continue
            
        # Calculate metrics
        if 'cumulative_rewards' in results and results['cumulative_rewards']:
            final_reward = results['cumulative_rewards'][-1]
            max_reward = max(results['cumulative_rewards'])
            
            analysis[model]['final_reward'] = final_reward
            analysis[model]['max_reward'] = max_reward
            analysis[model]['rewards'] = results['cumulative_rewards']
            
            logger.info(f"Model {model}: Final reward = {final_reward}, Max reward = {max_reward}")
    
    return analysis

def plot_results(analysis, output_dir):
    """Generate plots from the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot reward curves for each model
    plt.figure(figsize=(12, 6))
    for model, data in analysis.items():
        if 'rewards' in data:
            rewards = data['rewards']
            plt.plot(rewards, label=model)
    
    plt.title('Cumulative Rewards for AutonomousExcavatorGame')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cumulative_rewards.png'))
    plt.close()
    
    # Create a bar chart of final rewards for all models
    models = list(analysis.keys())
    final_rewards = [analysis[model]['final_reward'] for model in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, final_rewards, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Final Reward')
    plt.title('Final Rewards by Model for AutonomousExcavatorGame')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_rewards.png'))
    plt.close()
    
    # Save the analysis as JSON
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        analysis_dict = {model: data for model, data in analysis.items()}
        
        # Remove the rewards arrays to keep the JSON file small
        for model in analysis_dict:
            if 'rewards' in analysis_dict[model]:
                del analysis_dict[model]['rewards']
                
        json.dump(analysis_dict, f, indent=2)

def main():
    """Main function."""
    args = parse_args()
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(args.input_dir, args.models)
    
    if not experiment_dirs:
        logger.error(f"No experiment directories found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    # Analyze results
    analysis = analyze_results(experiment_dirs)
    
    # Plot results
    plot_results(analysis, args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
#!/bin/bash

# Parallel LLM Experiment Runner
# Usage: ./run_parallel_experiments.sh [options]

# Set default values
MODEL_NAME="gemini-2.5-pro-preview-05-06"
MODEL_KEY="gemini"
NUM_TIMESTEPS=120
SEED=58
NUM_ENVS=4
NUM_PROCESSES=""
RUN_NAME="/home/gioelemo/Documents/terra/no-action-map.pkl"
ENABLE_RENDERING=true

# Set environment variables
export DATASET_PATH=/home/gioelemo/Documents/terra/data/terra/train
export DATASET_SIZE=10

# Function to display help
show_help() {
    cat << EOF
Parallel LLM Experiment Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -m, --model-name MODEL          LLM model name (default: $MODEL_NAME)
    -k, --model-key KEY            Model key (gpt/gemini/claude) (default: $MODEL_KEY)
    -t, --timesteps NUM            Number of timesteps (default: $NUM_TIMESTEPS)
    -s, --seed NUM                 Random seed (default: $SEED)
    -n, --num-envs NUM             Number of environments/experiments (default: $NUM_ENVS)
    -p, --processes NUM            Number of parallel processes (default: auto-detect)
    -r, --run-name PATH            Path to checkpoint file (default: $RUN_NAME)
    --enable-rendering             Enable rendering (disabled by default for speed)
    -h, --help                     Show this help message

EXAMPLES:
    # Run 8 experiments in parallel with default settings
    $0 -n 8
    
    # Run with different model and more timesteps
    $0 -m "gpt-4o" -k "gpt" -t 200 -n 4
    
    # Run with specific number of processes
    $0 -n 8 -p 4
    
    # Run with rendering enabled (slower but visual)
    $0 -n 2 --enable-rendering

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -k|--model-key)
            MODEL_KEY="$2"
            shift 2
            ;;
        -t|--timesteps)
            NUM_TIMESTEPS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -n|--num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        -p|--processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        -r|--run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --enable-rendering)
            ENABLE_RENDERING=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate model combinations
case $MODEL_KEY in
    "gpt")
        case $MODEL_NAME in
            "gpt-4o"|"gpt-4.1"|"o4-mini"|"o3"|"o3-mini")
                ;;
            *)
                echo "Error: Invalid model name '$MODEL_NAME' for model key '$MODEL_KEY'"
                exit 1
                ;;
        esac
        ;;
    "gemini")
        case $MODEL_NAME in
            "gemini-1.5-flash-latest"|"gemini-2.0-flash"|"gemini-2.5-pro-exp-03-25"|"gemini-2.5-pro-preview-03-25"|"gemini-2.5-pro-preview-05-06"|"gemini-2.5-flash-preview-04-17"|"gemini-2.5-flash-preview-05-20")
                ;;
            *)
                echo "Error: Invalid model name '$MODEL_NAME' for model key '$MODEL_KEY'"
                exit 1
                ;;
        esac
        ;;
    "claude")
        case $MODEL_NAME in
            "claude-3-haiku-20240307"|"claude-3-7-sonnet-20250219"|"claude-opus-4-20250514"|"claude-sonnet-4-20250514")
                ;;
            *)
                echo "Error: Invalid model name '$MODEL_NAME' for model key '$MODEL_KEY'"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Error: Invalid model key '$MODEL_KEY'. Must be one of: gpt, gemini, claude"
        exit 1
        ;;
esac

# Check if checkpoint file exists
if [[ ! -f "$RUN_NAME" ]]; then
    echo "Error: Checkpoint file '$RUN_NAME' not found"
    exit 1
fi

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAFE_MODEL_NAME=${MODEL_NAME//\//_}
EXPERIMENT_DIR="experiments/parallel_${SAFE_MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

# Build the command
CMD="python -m llm.main_llm_parallel"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --model_key $MODEL_KEY"
CMD="$CMD --num_timesteps $NUM_TIMESTEPS"
CMD="$CMD -s $SEED"
CMD="$CMD -n $NUM_ENVS"
CMD="$CMD -run $RUN_NAME"

if [[ -n "$NUM_PROCESSES" ]]; then
    CMD="$CMD --num_processes $NUM_PROCESSES"
fi

if [[ "$ENABLE_RENDERING" == "true" ]]; then
    CMD="$CMD --enable_rendering"
fi

# Display configuration
echo "=========================================="
echo "Parallel LLM Experiment Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME ($MODEL_KEY)"
echo "Timesteps: $NUM_TIMESTEPS"
echo "Seed: $SEED"
echo "Number of experiments: $NUM_ENVS"
if [[ -n "$NUM_PROCESSES" ]]; then
    echo "Parallel processes: $NUM_PROCESSES"
else
    echo "Parallel processes: auto-detect"
fi
echo "Checkpoint: $RUN_NAME"
echo "Rendering enabled: $ENABLE_RENDERING"
echo "Dataset path: $DATASET_PATH"
echo "Dataset size: $DATASET_SIZE"
echo "Output directory: $EXPERIMENT_DIR"
echo "=========================================="

# Confirm execution
read -p "Do you want to proceed with this configuration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 0
fi

# Log the command and configuration
echo "Command: $CMD" > "$EXPERIMENT_DIR/command.txt"
echo "Configuration:" >> "$EXPERIMENT_DIR/config.txt"
echo "Model: $MODEL_NAME ($MODEL_KEY)" >> "$EXPERIMENT_DIR/config.txt"
echo "Timesteps: $NUM_TIMESTEPS" >> "$EXPERIMENT_DIR/config.txt"
echo "Seed: $SEED" >> "$EXPERIMENT_DIR/config.txt"
echo "Number of experiments: $NUM_ENVS" >> "$EXPERIMENT_DIR/config.txt"
echo "Parallel processes: $NUM_PROCESSES" >> "$EXPERIMENT_DIR/config.txt"
echo "Rendering enabled: $ENABLE_RENDERING" >> "$EXPERIMENT_DIR/config.txt"
echo "Start time: $(date)" >> "$EXPERIMENT_DIR/config.txt"

# Run the experiment
echo "Starting parallel experiments..."
echo "Logs will be saved to: $EXPERIMENT_DIR/experiment.log"

# Run with output redirection
$CMD 2>&1 | tee "$EXPERIMENT_DIR/experiment.log"

# Record completion
echo "End time: $(date)" >> "$EXPERIMENT_DIR/config.txt"

echo "Experiment completed. Check $EXPERIMENT_DIR for results."
# Game with LLMs

## Installation

Instead of using the `environment.yml` use the `environment_llm.yml` in the `feature/llm` branch

You also need an OPENAI API (`OPENAI_API_KEY.txt`) and a GOOGLE API `GOOGLE_API_KEY_FREE.txt` keys stored into the root folder of `terra`

## Execution
The code can be executed with the following command

```bash
DATASET_PATH=<PATH_TO_DATSET> DATASET_SIZE=<DATASET_SIZE> python -m terra.viz.main_llm -model_name <MODEL_NAME> --model_key <MODEL_KEY> --num_timesteps <NUM_TIMESTEPS>
```
Where
- `<PATH_TO_DATSET>` is the path where the maps are currently stored
- `<DATASET_SIZE>` is the number of elements in the dataset
- `<MODEL_NAME>` is the name of the model:
    * "gpt-4o", 
    * "gpt-4.1",
    * "o4-mini",
    * "o3",
    * "gemini-1.5-flash-latest",
    * "gemini-2.0-flash", 
    * "gemini-2.5-pro-exp-03-25"
    * "gemini-2.5-pro-preview-03-25", 
    * "gemini-2.5-flash-preview-04-17",
    * "claude-3-haiku-20240307", 
    * "claude-3-7-sonnet-20250219"
- `<MODEL_KEY>` is the key of the model
    * "gpt", 
    * "gemini"
    * "claude"
- `<NUM_TIMESTEPS>` is the number of timesteps (default value is 100)

For example a possible command should be:

```bash
DATASET_PATH=/home/gioelemo/Documents/terra/data/terra/train DATASET_SIZE=100 python -m terra.viz.main_llm --model_name gemini-2.5-pro-exp-03-25 --model_key gemini --num_timesteps 100 
```

## References
This implementation is higly based on [Atari-GPT: Investigating the Capabilities of Multimodal Large Language Models as Low-Level Policies for Atari Games](https://github.com/nwayt001/atari-gpt)
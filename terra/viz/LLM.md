# Game with LLMs

## Installation

Instead of using the `environment.yml` use the `environment_llm.yml` in the `feature/llm` branch

You also need an OPENAI API key stored into `OPENAI_API_KEY.txt` in the root folder of terra

## Execution
The code can be executed with the following command

`DATASET_PATH=<PATH_TO_DATSET> DATASET_SIZE=<DATASET_SIZE> python -m terra.viz.main_manual_llm`

## References
This implementation is higly based on [Atari-GPT: Investigating the Capabilities of Multimodal Large Language Models as Low-Level Policies for Atari Games](https://github.com/nwayt001/atari-gpt)
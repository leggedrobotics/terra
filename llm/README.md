How to run the code

```python
time DATASET_PATH=/home/gioelemo/Documents/terra/data/terra/train DATASET_SIZE=10 python -m llm.main_llm --model_name gemini-2.5-pro-preview-05-06 --model_key gemini --num_timesteps 120 -s 58
```
Structure of the folder

```
- llm
    - api_keys
        - ANTHROPIC_API_KEY.txt
        - GOOGLE_API_KEY.txt
        - OPENAI_API_KEY.txt
    - prompts
        - delegation_decision.txt
        - excavator_action.txt
        - excavator_llm_advanced.txt
        - excavator_llm_simple.txt
        - master_partitioning.txt
    - __init__.py
    - ...
```
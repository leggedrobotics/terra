from pathlib import Path
import json

class PromptManager:
    """Lightweight prompt manager that loads from external files."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        self._cache = {}
    
    def get(self, prompt_name: str, **kwargs) -> str:
        """Get a prompt with optional variable substitution."""
        if prompt_name not in self._cache:
            self._load_prompt(prompt_name)
        
        prompt = self._cache[prompt_name]
        if kwargs:
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable {e} for prompt '{prompt_name}'")
        return prompt
    
    def _load_prompt(self, prompt_name: str):
        """Load a prompt from file."""
        # Try .txt file first
        txt_file = self.prompts_dir / f"{prompt_name}.txt"
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                self._cache[prompt_name] = f.read().strip()
            return
        
        # Try .json file
        json_file = self.prompts_dir / f"{prompt_name}.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'prompt' in data:
                    self._cache[prompt_name] = data['prompt']
                elif isinstance(data, str):
                    self._cache[prompt_name] = data
                else:
                    raise ValueError(f"Invalid JSON format in {json_file}")
            return
        
        raise FileNotFoundError(f"Prompt file not found: {prompt_name}.txt or {prompt_name}.json")
    
    def reload(self, prompt_name: str = None):
        """Reload prompts from files (useful for development)."""
        if prompt_name:
            if prompt_name in self._cache:
                del self._cache[prompt_name]
        else:
            self._cache.clear()
import ast
from typing import List, Tuple

def parse_tasks(text: str) -> List[Tuple[str, str, List[str]]]:
    """
    Parse a pipe-separated task breakdown into a List of (name, instruction, keys).
    """
    text = _strip_fences(text)
    tasks: List[Tuple[str, str, List[str]]] = []
    
    for line in text.strip().splitlines():
        # Skip empty lines
        if not line.strip():
            continue
        
        # Split by '|' and strip whitespace
        parts = [part.strip() for part in line.split('|')]
        if len(parts) != 3:
            raise ValueError(f"Line does not have exactly 3 parts: {line!r}")
        
        name, instruction, keys_str = parts
        
        # Safely parse the keys list literal
        try:
            keys = ast.literal_eval(keys_str)
        except Exception as e:
            raise ValueError(f"Failed to parse keys on line: {line!r}\n  {e}")
        
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError(f"Parsed keys is not a list of strings: {keys!r}")
        
        tasks.append((name, instruction, keys))
    
    return tasks

def _strip_fences(code: str) -> str:
    # Detect triple backtick fenced code block and extract inner content
        if "```" in code:
            lines = code.splitlines()
            inside_block = False
            code_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    inside_block = not inside_block
                    continue
                if inside_block:
                    code_lines.append(line)
            return "\n".join(code_lines).strip()
        return code.strip()

if __name__ == "__main__":
    data = """
    Based on the configuration, I will outline a high-level design for the game. The necessary classes and their corresponding tasks are as follows:

```
GameManager | Manage game state and progress | ['game_name', 'game_mechanics', 'objectives']
CharacterController | Handle character movement and actions | ['characters', 'controls']
PlatformGenerator | Generate platforms for the game world | ['setting', 'dimension']
ObjectManager | Manage collectible and non-collectible objects | ['objects']
ScoringSystem | Handle scoring and win conditions | ['objectives', 'scoring']
Camera | Manage the camera view and rendering | ['camera_view', 'dimension']
EffectManager | Handle visual effects such as sparkle effects | ['objects']
CollisionDetector | Detect collisions between the character and game objects | ['objects', 'characters']
```

Here's a brief explanation of each class:
    """
    
    tasks = parse_tasks(data)
    print(tasks)

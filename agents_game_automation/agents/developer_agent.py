# agents/developer_agent.py

import os
import json
from typing import Dict, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv

class DeveloperAgent:
    """
    Generates modular Python/Pygame game code from a JSON config,
    creating one file per feature and maintaining a brief summary
    to ensure consistency across generated modules.
    """

    def __init__(
        self,
        model: str = "meta/llama-4-maverick-17b-128e-instruct",
        config_path: str = "game_config/game_config.json",
        output_dir: str = "generated_game"
    ):
        load_dotenv()
        api_key = os.getenv("META_LLAMA4")
        if not api_key:
            raise ValueError("META_LLAMA4 not set in .env")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = model
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # A brief summary of generated modules
        self.summary: List[str] = []

    def _llm(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert Python/Pygame game developer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    def _strip_fences(self, code: str) -> str:
        # Remove markdown ``` fences if present
        if code.startswith("```"):
            parts = code.split("```")
            # If language is specified, skip first part
            if len(parts) >= 3:
                return parts[1].split("\n", 1)[1]
            return parts[1]
        return code

    def _write_file(self, filename: str, code: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Wrote {filename}")

    def write_all(self):
        """
        Generate all game modules in sequence, each focused on a
        specific config slice, updating summary to keep consistency.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict = json.load(f)

        # Define tasks: (filename, high-level instruction, needed config keys)
        tasks: List[Tuple[str, str, List[str]]] = [
            (
                "main.py",
                "Initialize pygame, set window title to game_name, size 800x600, and create main loop.",
                ["game_name", "setting", "controls"]
            ),
            (
                "characters.py",
                "Define Character base class and concrete classes for each character in config.",
                ["characters"]
            ),
            (
                "controls.py",
                "Map key inputs to character movement based on controls mapping.",
                ["controls"]
            ),
            (
                "mechanics.py",
                "Implement game mechanics and objectives such as scoring, collisions, and win conditions.",
                ["mechanics", "objectives"]
            ),
            (
                "assets_loader.py",
                "Load and provide access to images listed in assets_required.",
                ["assets_required"]
            ),
            (
                "sounds.py",
                "Load and manage sounds from sounds_required, providing play_sound function.",
                ["sounds_required"]
            ),
            (
                "play.py",
                "This file will have function to play game and update score.",
                []
            ),
            (
                "main.py",
                "Update main.py to interact with all generated files and load the game.",
                []
            ),
        ]

        for filename, instruction, keys in tasks:
            # Build minimal subconfig for this module
            subconfig = {k: self.config.get(k) for k in keys}

            # Build prompt with summary and subconfig
            prompt = (
                "# Previously generated modules summary:\n" +
                ("\n".join(self.summary) if self.summary else "- none") +
                "\n\n" +
                f"Generate `{filename}`\nInstruction: {instruction}" +
                f"\nUse only this part of the config: {json.dumps(subconfig)}"
                "\nOutput only valid Python code."
            )

            # Call LLM and clean code
            raw_code = self._llm(prompt)
            code = self._strip_fences(raw_code)

            # Save file and update summary
            self._write_file(filename, code)
            self.summary.append(f"- {filename}: {instruction}")

# Runner
if __name__ == "__main__":
    agent = DeveloperAgent()
    agent.write_all()

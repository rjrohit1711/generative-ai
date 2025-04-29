# agents/developer_agent.py

import os
import shutil

class DeveloperAgent:
    def __init__(self):
        self.build_dir = "outputs/builds"

    def integrate_assets(self, game_description, asset_paths, sound_paths):
        """
        Create a minimal game package:
        - Copy assets & sounds into a build folder
        - Generate a simple runner script (e.g. Pygame stub) that loads them
        """
        print("🛠️  Integrating assets into game build...")
        os.makedirs(self.build_dir, exist_ok=True)

        # Copy files
        for path in asset_paths + sound_paths:
            shutil.copy(path, self.build_dir)

        # Create a stub runner
        runner_py = os.path.join(self.build_dir, "run_game.py")
        with open(runner_py, "w") as f:
            f.write(
                'import os\n'
                'print("Starting game with assets:", os.listdir("."))\n'
                '# Here you could import pygame and load images/sounds\n'
            )

        return self.build_dir

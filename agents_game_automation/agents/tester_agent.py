# agents/tester_agent.py

import subprocess
import os

class TesterAgent:
    def __init__(self):
        pass

    def test_game(self, build_path):
        """
        Run the runner script and capture output/error.
        Returns a simple pass/fail report.
        """
        print("🧪 Testing game build...")
        runner = os.path.join(build_path, "run_game.py")
        try:
            result = subprocess.run(
                ["python", runner],
                cwd=build_path,
                capture_output=True,
                text=True,
                check=True
            )
            report = f"✅ Game ran successfully:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            report = f"❌ Game test failed:\n{e.stderr}"
        return report

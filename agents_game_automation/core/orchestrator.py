# orchestrator.py

from agents.game_designer_agentv2 import GameDesignerAgent
from agents.asset_creator_agent import AssetCreatorAgent
from agents.sound_designer_agent import SoundDesignerAgent
from agents.developer_agent import DeveloperAgent
from agents.tester_agent import TesterAgent

class Orchestrator:
    def __init__(self):
        self.designer  = GameDesignerAgent()
        self.assets    = AssetCreatorAgent()
        self.sounds    = SoundDesignerAgent()
        self.dev       = DeveloperAgent()
        self.tester    = TesterAgent()

    def run(self, prompt=None):
        print("🎮 Orchestration started!")

        # 1. Config generation
        # self.designer.generate_game_config(prompt)

        # 2. Asset generation
        self.assets.generate_assets()

        # 3. Sound generation
        self.sounds.generate_sounds()

        # 4. Build integration
        build_dir = self.dev.write_all()

        # 5. Automated testing
        # report = self.tester.test_game(build_dir)

        # 6. Summary
        print("\n✅ All done!")

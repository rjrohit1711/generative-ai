# orchestrator.py

from agents.game_designer_agent import GameDesignerAgent
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

        # 1. Game design
        game_desc = self.designer.generate_game_description(prompt)

        # 2. Asset generation
        assets = self.assets.generate_assets(game_desc)

        # 3. Sound generation
        sounds = self.sounds.generate_sounds(game_desc)

        # 4. Build integration
        build_dir = self.dev.integrate_assets(game_desc, assets, sounds)

        # 5. Automated testing
        report = self.tester.test_game(build_dir)

        # 6. Summary
        print("\n✅ All done!")
        print(f"📝 Design:\n{game_desc}\n")
        print(f"🎨 Assets:\n{assets}\n")
        print(f"🎵 Sounds:\n{sounds}\n")
        print(f"📦 Build directory: {build_dir}")
        print(f"🧪 Test report:\n{report}")
        return {
            "design": game_desc,
            "assets": assets,
            "sounds": sounds,
            "build": build_dir,
            "test_report": report
        }

# orchestrator.py

from agents.game_designer_agentv2 import GameDesignerAgent
from agents.asset_creator_agent import AssetCreatorAgent
from agents.sound_designer_agent import SoundDesignerAgent
from agents.developer_agentv2 import DeveloperAgent
from agents.tester_agent import TesterAgent
import utils.constants as Constants

game_config = Constants.GAME_CONFIGV2

class Orchestrator:
    def __init__(self):
        self.designer  = GameDesignerAgent(game_config)
        self.assets    = AssetCreatorAgent(game_config)
        self.sounds    = SoundDesignerAgent(game_config)
        self.dev       = DeveloperAgent(game_config)
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
        self.dev.write_all()
        
        # 6. Summary
        print("\n✅ All done!")

import json

class GameDesignParser:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def get_game_name(self):
        return self.data.get("game_name", "")

    def get_art_style(self):
        return self.data.get("art_style", "")

    def get_setting(self):
        return self.data.get("setting", "")

    def get_character_list(self):
        return self.data.get("characters", [])

    def get_mechanics_list(self):
        return self.data.get("mechanics", [])

    def get_controls(self):
        return self.data.get("controls", {})

    def get_objectives(self):
        return self.data.get("objectives", [])

    def get_ui_elements(self):
        assets = self.data.get("assets_required", {})
        return assets.get("ui_elements", [])

    def get_backgrounds(self):
        assets = self.data.get("assets_required", {})
        return assets.get("backgrounds", [])

    def get_assets_characters(self):
        assets = self.data.get("assets_required", {})
        return assets.get("characters", [])

    def get_background_music_tracks(self):
        sounds = self.data.get("sounds_required", {})
        return sounds.get("background_music", [])

    def get_sound_effects(self):
        sounds = self.data.get("sounds_required", {})
        return sounds.get("sound_effects", [])

# Example usage
if __name__ == "__main__":
    parser = GameDesignParser("game_config_sample.json")
    
    print("Game Name:", parser.get_game_name())
    print("Art Style:", parser.get_art_style())
    print("Setting:", parser.get_setting())
    print("Characters:", parser.get_character_list())
    print("Mechanics:", parser.get_mechanics_list())
    print("Controls:", parser.get_controls())
    print("Objectives:", parser.get_objectives())
    print("UI Elements:", parser.get_ui_elements())
    print("Backgrounds:", parser.get_backgrounds())
    print("Assets Characters:", parser.get_assets_characters())
    print("Background Music Tracks:", parser.get_background_music_tracks())
    print("Sound Effects:", parser.get_sound_effects())

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

class GameDesignerAgent:
    def __init__(
        self,
        base_model: str = "meta/llama-4-maverick-17b-128e-instruct"
    ):
        load_dotenv()
        api_key = os.getenv("META_LLAMA4")
        if not api_key:
            raise ValueError("META_LLAMA4 environment variable is not set.")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model_name = base_model
        # lora_path is ignored because NVIDIA's endpoint handles the model — but keeping it for compatibility
        print(f"[LOG] Initialized GameDesignerAgentV2 with model {self.model_name}")

    def generate_game_config(self, concept: str) -> str:
        
        # Read the JSON file
        with open(r'dataset/game_design_data.json', 'r') as file:
          json_content = json.load(file)
          game_structure = """
            {
               "game_name": "<game_name>",
                "art_style": "<art_style>",
                "setting": "<setting>",
                "characters": [
                {
                    "name": "<character_name>",
                    "controls": {
                    "left": "left",
                    "right": "right",
                    "jump": "space"
                        }
                    }
                ],
                "objects": [
                    {
                    "name": "<object_name>",
                    "description": "<what it looks like or does>",
                    "collectible": true
                    }
                ],
                "game_mechanics": [
                "<mechanic_1>",
                "<mechanic_2>",
                "<mechanic_3>"
                ],
                "assets_required": {
                "<asset_1>": { "description": "<description>" },
                "<asset_2>": { "description": "<description>" },
                "<asset_3>": { "description": "<description>" },
                ...
                },
                "sounds_required": {
                "<sound_1>": { "description": "<description>" },
                "<sound_2>": { "description": "<description>" },
                "<sound_3>": { "description": "<description>" },
                ...
                }
            }
            """
        
        prompt =  f"""
            You are an expert game designer. 
            Design a new video game based on the following idea:

            Idea: {concept}

            Output the game design strictly in the following JSON format, don't add any extra column:
            {game_structure}
           
            Keep the overall game setting very simple.

            // Use this as reference : {json.dumps(json_content)}
            """


        completion = self.client.chat.completions.create(
            model=self.model_name,
              messages=[
                {"role": "system", "content": "You are an expert game designer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            top_p=0.8,
            max_tokens=1024,
            stream=True
        )
        print(completion)
        # Collect streamed chunks into full text
        full_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_content += chunk.choices[0].delta.content

        # Now extract only the JSON part
        json_start = full_content.find("{")
        if json_start == -1:
            raise ValueError("No JSON object found in the output.")

        json_part = full_content[json_start:]

         # Balance braces to cut off extra text
        depth = 0
        for idx, ch in enumerate(json_part):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                final_json = json_part[: idx + 1]
                # Save to file
                with open("bin/source/game_config.json", "w", encoding="utf-8") as f:
                    f.write(final_json)
                return final_json

        # If brace balancing fails, still save whatever we have
        with open("bin/source/game_config.json", "w", encoding="utf-8") as f:
            f.write(json_part)
        return json_part

# Example usage
if __name__ == "__main__":
    agent = GameDesignerAgent(
        base_model="google/gemma-3-1b-it"
    )
    concept = "I want to build a game where dragon fly and collect coins."
    print("\n=== Generated Game Design ===\n")
    print(agent.generate_game_config(concept))

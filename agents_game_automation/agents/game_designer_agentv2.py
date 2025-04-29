import os
import json
from openai import OpenAI
from dotenv import load_dotenv

class GameDesignerAgent:
    def __init__(
        self,
        base_model: str = "google/gemma-3-1b-it"
    ):
        load_dotenv()
        api_key = os.getenv("GEMMA3B_API_KEY")
        if not api_key:
            raise ValueError("GEMMA3B_API_KEY environment variable is not set.")

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
        
        prompt =  f"""
        You are an expert game designer. 
        Design a new video game based on the following idea:

        Idea: {concept}

        Output the game design strictly in the following JSON format, dont add any extra column:
        {{
        "game_name": "<game_name>",
        "art_style": "<art_style>",
        "setting": "<setting>",
        "characters": ["<character1>", "<character2>", "..."],
        "mechanics": ["<mechanic1>", "<mechanic2>", "..."],
        "controls": {{"<control1>": "<key>", "<control2>": "<key>"}},
        "objectives": ["<objective1>", "<objective2>", "..."],
        "assets_required": {{"<asset_type>": ["<asset1>", "<asset2>"]}},
        "sounds_required": {{"<sound_type>": ["<sound1>", "<sound2>"]}}
        }}

        Use this as reference : {json.dumps(json_content)}
        """

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            top_p=0.8,
            max_tokens=600,
            stream=True
        )

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
                with open("game_config/game_config.json", "w", encoding="utf-8") as f:
                    f.write(final_json)
                return final_json

        # If brace balancing fails, still save whatever we have
        with open("game_config/game_config.json", "w", encoding="utf-8") as f:
            f.write(json_part)
        return json_part

# Example usage
if __name__ == "__main__":
    agent = GameDesignerAgent(
        base_model="google/gemma-3-1b-it"
    )
    concept = "A cozy Mario racing game in a haunted library."
    print("\n=== Generated Game Design ===\n")
    print(agent.generate_game_config(concept))

from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
import json

# Read the JSON file
with open(r'dataset/game_design_data.json', 'r') as file:
    json_content = json.load(file)
concept = "A cozy racing game in a haunted library."
completion = client.chat.completions.create(
    model="google/gemma-3-1b-it",
    messages=[{"role": "user", "content":
    f"""
        You are an expert game designer. 
        Design a new video game based on the following idea:

        Idea: {concept}

        Output the game design in the following JSON format:
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

        Use this as reference: {json.dumps(json_content)}
        """
    }],
  temperature=0.9,
  top_p=0.8,
  max_tokens=700,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


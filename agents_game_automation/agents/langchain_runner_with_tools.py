import os
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Load env vars
load_dotenv()
# (LangChain’s ChatOpenAI will read OPENAI_* from env under the hood)

# 2. Instantiate the LLM (Qwen via NVIDIA Integrate)
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL"),
    openai_api_key=os.getenv("QWEN_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.0)),
)

from langchain.agents import initialize_agent, AgentType

from langchain.agents import Tool

# Dummy functions
def dummy_recipe_fetcher(input: str) -> str:
    return f"Here are dummy recipes based on ingredients: {input}\n1. Dummy Pizza\n2. Dummy Pasta"

def dummy_weather_fetcher(location: str) -> str:
    return f"Dummy weather for {location}: 25°C, clear sky."

# Tool definitions
fetch_recipes_tool = Tool(
    name="RecipeFetcher",
    func=dummy_recipe_fetcher,
    description="Useful for fetching recipes based on a list of ingredients. Input should be a comma-separated list of ingredients."
)

fetch_weather_tool = Tool(
    name="WeatherFetcher",
    func=dummy_weather_fetcher,
    description="Given a query like 'city' or 'tomorrow', returns dummy weather."
)

# Optional: list of tools for use in an agent
tools = [fetch_recipes_tool, fetch_weather_tool]

# 3) Create SQLite memory
memory =  ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,   # ← enable automatic retry on parse errors
    memory=memory,
    verbose=True
)

response1 = agent.run("What’s the weather in tanakpur?")

response2 = agent.run("And what about tomorrow's weather?")  
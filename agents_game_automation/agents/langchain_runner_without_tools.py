import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory 

# 1. Load env vars
load_dotenv()

# 2. Setup LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    openai_api_key=os.getenv("QWEN_API_KEY"),  
    openai_api_base=os.getenv("OPENAI_API_BASE"), 
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.0))
)

# 3. Define prompt with history
def get_conversation_prompt(messages):
    # Create a prompt with the full message history
    history_prompt = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            history_prompt += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            history_prompt += f"AI: {message.content}\n"
    return history_prompt + "Human: {input}\n"  # Add the new user input

# 4. In-memory store for sessions
store = {}

# 5. Message history factory
def get_message_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 6. Chain with proper message history
def chain_with_history(session_id, input_message):
    history = get_message_history(session_id)
    history.add_user_message(input_message)
    
    # Generate the prompt from the history
    prompt = get_conversation_prompt(history.messages)
    
    # Call the model with the prompt (pass it as a string now)
    response = llm.invoke(prompt)
    
    # Ensure that response is an AIMessage and get the 'text' attribute
    if isinstance(response, AIMessage):
        ai_response_text = response.content
    else:
        ai_response_text = response
    
    # Add AI response to the history
    history.add_ai_message(ai_response_text)

    return ai_response_text

# 7. Simulate a conversation
session_id = "rohit-session"

print(chain_with_history(session_id, "Hi, I'm Rohit."))
print(chain_with_history(session_id, "What did I just tell you?"))
print(chain_with_history(session_id, "Remember that I'm a developer."))
print(chain_with_history(session_id, "What do you know about me now?"))

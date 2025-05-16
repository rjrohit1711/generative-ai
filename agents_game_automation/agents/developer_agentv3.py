# agents/developer_agent.py

import os, sys, ast
from dotenv import load_dotenv
import json
from typing import Dict, List, Tuple


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.constants as Constants
from agents.tester_agent import TesterAgent

class DeveloperAgent:
    """
    Generates modular Python/Pygame game code from a JSON config,
    creating one file per feature and maintaining a brief summary
    to ensure consistency across generated modules.
    """

    def __init__(
        self,
        config_path: str = Constants.GAME_CONFIGV3,
        output_dir: str = "bin/source/gamev3"
    ):
        load_dotenv()
        api_key = os.getenv("LAMA_70")
        if not api_key:
            raise ValueError("LAMA_70 not set in .env")

        self.client = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            openai_api_key=api_key,  
            openai_api_base=os.getenv("OPENAI_API_BASE"), 
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.2))
        )
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.store = {}

    def _llm(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response.content.strip()

    def get_conversation_prompt(self, messages):
        # Create a prompt with the full message history
        history_prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                history_prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                history_prompt += f"AI: {message.content}\n"
        return history_prompt + "Human: {input}\n"

    def get_message_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _strip_fences(self, code: str) -> str:
    # Detect triple backtick fenced code block and extract inner content
        if "```" in code:
            lines = code.splitlines()
            inside_block = False
            code_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    inside_block = not inside_block
                    continue
                if inside_block:
                    code_lines.append(line)
            return "\n".join(code_lines).strip()
        return code.strip()
    
    def _ask_yes_no(self, prompt: str) -> bool:
        while True:
            answer = input(f"{prompt} (y/n): ").strip().lower()
            if answer in ['y', 'yes']:
                return True
            elif answer in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'.")

    def parse_tasks(self, text: str) -> List[Tuple[str, str, List[str]]]:
        """
        Parse a pipe-separated task breakdown into a List of (name, instruction, keys).
        """
        text = self._strip_fences(text)
        tasks: List[Tuple[str, str, List[str]]] = []
        
        for line in text.strip().splitlines():
            # Skip empty lines
            if not line.strip():
                continue
            
            # Split by '|' and strip whitespace
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 3:
                raise ValueError(f"Line does not have exactly 3 parts: {line!r}")
            
            name, instruction, keys_str = parts
            
            # Safely parse the keys list literal
            try:
                keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            except Exception as e:
                raise ValueError(f"Failed to parse keys on line: {line!r}\n  {e}")
            
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise ValueError(f"Parsed keys is not a list of strings: {keys!r}")
            
            tasks.append((name, instruction, keys))
        
        return tasks

    def _write_file(self, filename: str, code: str):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Wrote {filename}")

    def write_all(self):
        """
        Generate all game modules IN sequence, each focused on a
        specific config slice, updating summary to keep consistency.
        Make sure to write full code don't assume code will be there.
        Make sure each class has very minimal dependency to other classes if it need any config data it should have one.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict = json.load(f)

        tasks, summary_str = self._get_tasks()
        if(tasks is None):
            return
        
        for i in range(1):
            self._code(summary_str, tasks, i)
    
    def _get_tasks(self):
        task_summary = None
        # Define tasks: (filename, high-level instruction, needed config keys)
        for _ in range(2):
            task_prompt = f"""
                You are a game developer assistant. I will provide a game configuration file in JSON format. Based on this config, your task is to design a Pygame game.
                - **Game configuration file** (`game_config.json`):  
                {self.config}

                Your output must follow the structure shown below and strictly adhere to this format:
                Tasks:
                filename.py | A  Detailed functionality of that class/module and also mention its dependencies on other classes. | List of relevant config keys (from the JSON)
                ...
             
                - Use `|` as a separator.
                - Do **not** include any explanation, code, or markdown headings — just the block enclosed in triple backticks as shown above.
 
                ### Structure Rules:

                1. **Start with individual component classes** — These should be low-level, focused on specific responsibilities.
                - Make sure they expose expected functionalities so that other classes can integrate them.
                - Make as small class as possible for each objects, character or entities in the game.

                2. **Then define integration classes** — These should tie the system together.

                ### Input:

                - **Previously rejected summary**:  
                {task_summary}
                ---
                Now, output only the structured list of classes/modules as per the format.
                """
            
            task_report = self._llm(task_prompt)
            print(f"Report: {task_report}")
            summary_str = task_report

            tasks: List[Tuple[str, str, List[str]]] = self.parse_tasks(task_report)

            print(f"Tasks: {tasks}")
            task_summary = task_report
            if self._ask_yes_no("Do you want to continue?"):
                return tasks, summary_str
        
        return None, None
    
    def _code(self, summary_str, tasks, i):
        session_id = "rohit-session"
        history = self.get_message_history(session_id)
        history.add_user_message(str(self.config))
        history.add_user_message(summary_str)
    
        for filename, instruction, keys in tasks:       
            # Generate the prompt from the history
            summary_str = self.get_conversation_prompt(history.messages)
            # Build minimal subconfig for this module
            subconfig = {k: self.config.get(k) for k in keys}
            # Promt llm to generate class signature for given instruction and subconfig and capture results.
            stub_prompt = f"""

            Lets start to code.

            # SUMMARY SO FAR:
            { summary_str }

            TASK     : Create the STUB for `{filename}`
            INSTRUCTION: {instruction}
            CONFIG   : {json.dumps(subconfig, indent=2)}

            OUTPUT ONLY:
            - necessary imports
            - Docs
            - class or function signatures
            - `todo` in method bodies
            - Copy required config data as class constants individually, don't just copy them json objects.
            - Make sure to import and reuse already defined classes from the summary provided.
            """
            stub_code = self._llm(stub_prompt)
            print(f"STUB Code generated. {stub_code}")
            history.add_user_message(stub_code)

            # Prompt llm to write implementation.
            impl_prompt = f"""
            You have this stub in `{filename}`:
            Summary {summary_str}
            ```python
            {stub_code}
            CONFIG : {json.dumps(subconfig, indent=2)} INSTRUCTION: "{instruction}"
            - Please replace every `todo` with working implementation.
            - Remove doc strings, not required anymore.
            - Return the full, updated Python file only.
            - Make sure to import and reuse already defined classes from the summary provided.
            """
            impl_code = self._llm(impl_prompt)
            print(f"IMPL Code generated. {impl_code}")

            # Prompt llm to write main function.
            main_prompt = f"""
            You have this stub in `{filename}`:
            ```python
            {impl_code}"
            - Add "__main__" to run class independently as well for testing purposes.
            - Make sure not to add any infinite loop or human interaction in `main` call as AI will be testing this automatically, add a timeout of 10 seconds.
            """
            main_code = self._llm(main_prompt)
            print(f"IMPL Code generated. {main_code}")

            # Prompt llm to refine/check for errors.
            lint_prompt = f"""
            Your implementation of `{filename}` may have lint or syntax issues.  
            Run `flake8` and fix ANY errors.
            Remove unused variables and import.

            Here is the current code:
            ```python
            {main_code}
            Remove doc strings, not required anymore.
            Return the complete corrected file—do not alter logic, only style/syntax. """
            impl_code = self._llm(lint_prompt)
            print(f"LINT Code generated. {impl_code}")

            # Final iteration and save as done below 
            final_prompt = f"""
                All tests now pass for `{filename}`.  
                As a last step, please add a brief module‐level docstring at the top 
                summarizing its responsibility, then return the final file.
                {impl_code}
                """
            
            raw_code = self._llm(final_prompt)
            final_code = self._strip_fences(raw_code)
            self._write_file(filename, final_code)
            self._write_file("summary.txt", summary_str)

            # Prompt tester agent to write a test class and test it.
            TesterAgent().test_agent(os.path.join(self.output_dir, filename), summary_str, subconfig)

# Runner
if __name__ == "__main__":
    agent = DeveloperAgent()
    agent.write_all()

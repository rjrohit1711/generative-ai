# agents/developer_agent.py

import os, sys, ast
import json
from typing import Dict, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv

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
        model: str = "qwen/qwen2.5-coder-32b-instruct",
        config_path: str = Constants.GAME_CONFIGV3,
        output_dir: str = "bin/source/gamev3"
    ):
        load_dotenv()
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY not set in .env")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = model
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # A brief summary of generated modules
        self.summary: List[str] = []

    def _llm(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert Python/Pygame game developer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()

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
                keys = ast.literal_eval(keys_str)
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

                Previous Summary that was rejected by me - {task_summary}

                Here is the config file:
                game_config.json: {self.config}

                You are a game developer assistant. I will provide you with a game configuration in JSON format. Your task is to design a simple game using Pygame based on the data from this config file.

                `filename.py | A concise summary of that Class | List of related config keys (from the JSON)`

                Use `|` as a separator and use this format and make it enclosed in ```:
                \n
                ```
                str | str | List of [ str ]
                str | str | List of [ str ]
                str | str | List of [ str ]
                str | str | List of [ str ]
                str | str | List of [ str ]
                str | str | List of [ str ]
                ...
                ```
                Follow above format strictly to generate data.
                First define individual components, after that define components which will integrate them to make game.
                \n
                Do not generate any actual code yet—just the structured design and class breakdown. Just output the task entires.
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
        for filename, instruction, keys in tasks:
            # Build minimal subconfig for this module
            subconfig = {k: self.config.get(k) for k in keys}
            summary_str = '\n'.join(self.summary) if self.summary else '- none'
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
            """
            stub_code = self._llm(stub_prompt)
            print(f"STUB Code generated. {stub_code}")

            # Prompt llm to write implementation.
            impl_prompt = f"""
            You have this stub in `{filename}`:
            ```python
            {stub_code}
            CONFIG : {json.dumps(subconfig, indent=2)} INSTRUCTION: "{instruction}"
            Please replace every `todo` with working implementation.
            Remove doc strings, not required anymore.
            Return the full, updated Python file only. 
            """
            impl_code = self._llm(impl_prompt)
            print(f"IMPL Code generated. {impl_code}")

            # Prompt llm to refine/check for errors.
            lint_prompt = f"""
            Your implementation of `{filename}` may have lint or syntax issues.  
            Run `flake8` and fix ANY errors.

            Here is the current code:
            ```python
            {impl_code}
            Remove doc strings, not required anymore.
            Return the complete corrected file—do not alter logic, only style/syntax. """
            impl_code = self._llm(lint_prompt)
            print(f"LINT Code generated. {impl_code}")

            # Prompt tester agent to write a test class and test it.
            # test_results = TesterAgent().test_code(filename, impl_code)
            test_results = None

            # Provide feedback if encounter any errors to llm.
            if(test_results is not None):
                refine_prompt =  f"""
                Your tests for `{filename}` failed with this output:
                {test_results}

                Please correct only the parts of `{filename}` that cause these failures.  
                Return the full, fixed Python file.
                """
                impl_code = self._llm(refine_prompt)
                
            # Final iteration and save as done below 
            final_prompt = f"""
                All tests now pass for `{filename}`.  
                As a last step, please add a brief module‐level docstring at the top 
                summarizing its responsibility, then return the final file.
                {impl_code}
                """
            
            raw_code = self._llm(final_prompt)
            final_code = self._strip_fences(raw_code)
            if(i == 0):                
                self.summary.append(final_code)
            # Save file
            self._write_file(filename, final_code)

# Runner
if __name__ == "__main__":
    agent = DeveloperAgent()
    agent.write_all()

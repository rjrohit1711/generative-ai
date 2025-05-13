# agents/developer_agent.py

import os, sys
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
        config_path: str = Constants.GAME_CONFIGV2,
        output_dir: str = "bin/source/gamev2"
    ):
        load_dotenv()
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY not set in .env")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
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
            temperature=0.2,
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

    def _write_file(self, filename: str, code: str):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Wrote {filename}")

    def write_all(self):
        """
        Generate all game modules in sequence, each focused on a
        specific config slice, updating summary to keep consistency.
        Make sure to write full code don't assume code will be there.
        Make sure each class has very minimal dependency to other classes if it need any config data it should have one.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict = json.load(f)

        # Define tasks: (filename, high-level instruction, needed config keys)
        tasks: Tuple[str, str] =[
                "play.py",
                "Write the code to convert this config file to a working game.\n"
                "Do add a screen asking to start the game and add win/lose screen as well.\n"
                "Make sure to use sounds and sprites path provided for each entity."
                "Make small callable functions/classes.\n"
                "Make sure to check code for any error or assumable parameters.\n"
                "Make use of all assets and sound provided on case basis.\n"
                "Write simple easy to handle code.\n"
        ]
        
        self._code( tasks)

    def _code(self, task):
        filename, instruction = task
        config = self.config
        summary_str = '\n'.join(self.summary) if self.summary else '- none'
        # Promt llm to generate class signature for given instruction and subconfig and capture results.
        stub_prompt = f"""
            # SUMMARY SO FAR:
            { summary_str }

            TASK     : Create the STUB for `{filename}`
            INSTRUCTION: {instruction}
            CONFIG   : {json.dumps(config)}

            OUTPUT ONLY:
            - necessary imports.
            - class or function signatures.
            - `#todo` in method bodies.
            - Extract config data as individual and meaningful contants to use later in code.
            """
        stub_code = self._llm(stub_prompt)
        print(f"STUB Code generated. {stub_code}")


        # Prompt llm to write implementation.
        impl_prompt = f"""
        You have this stub in `{filename}`:
        ```python
        {stub_code}
        INSTRUCTION: "{instruction}"
        Please replace every `#todo` with working implementation.
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

        # Save file
        self._write_file(filename, final_code)
        self.summary.append(impl_code)
        self._write_file("summary.txt", summary_str)

# Runner
if __name__ == "__main__":
    agent = DeveloperAgent()
    agent.write_all()

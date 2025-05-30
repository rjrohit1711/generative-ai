# agents/developer_agent.py

import re
from inputimeout import inputimeout, TimeoutOccurred
import textwrap, subprocess
import os, sys, shutil
from dotenv import load_dotenv
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque

from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.constants as Constants
from agents.tester_agent import TesterAgent

Task = Tuple[str, str, List[str], List[str]]

class DeveloperAgent:
    """
    Generates modular Python/Pygame game code from a JSON config,
    creating one file per feature and maintaining a brief summary
    to ensure consistency across generated modules.
    """
    def __init__(
        self, 
    ):
        self.client = self._define_client()
        self.agent = self._agent()
        self.config_path = os.path.join(Constants.CONFIG_BASE, Constants.GAME_CONFIGV3)
        self.output_dir = "bin/source/gamev3"
        self.test_dir = os.path.join(self.output_dir, "tests")
        self.data_dir = os.path.join(self.output_dir, "data")
        self.tasks_data = os.path.join(self.data_dir, "tasks.txt")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def develop_game(self):
        """
        Generate all game modules IN sequence, each focused on a
        specific config slice, updating summary to keep consistency.
        Make sure to write full code don't assume code will be there.
        Make sure each class has very minimal dependency to other classes if it need any config data it should have one.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.manifest: Dict = json.load(f)

        tasks = self._get_tasks()
        if(tasks is None):
            return
        
        self._write_code_and_tests(tasks)

    def _define_client(self):
        load_dotenv()
        api_key = os.getenv("LAMA_70")
        if not api_key:
            raise ValueError("LAMA_70 not set in .env")
        
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            openai_api_key=api_key,  
            openai_api_base=os.getenv("OPENAI_API_BASE"), 
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.0))
        )
    
    def _agent(self):
        memory =  ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        write_unit_test_tool = Tool(
            name="write_unit_test",
            func=lambda filename: self._write_unit_test(filename),
            description=(
                "Generates a unit test class for a given class file. "
                "Input format: 'filename' "
                "(e.g. 'apple.py')"
            )
        )

        run_unit_test_tool = Tool(
            name="run_unit_test",
            func=lambda filename: self._run_unit_test(filename),
            description=(
                "Runs the unit test file for the given class and returns the result. "
                "Input format: 'test_filename' "
                "(e.g. 'Test_apple.py')"
            )
        )

        patch_code_from_test_tool = Tool(
            name="patch_code_from_test",
            func=lambda args: self._patch_code_from_test(args),
            description=(
                "Patches the class implementation based on the unit test failure output. "
                "Input format: 'filename||entire stack trace' "
                "(e.g. 'apple.py||<entire stack trace>')"
            )
        )

        human_feedback_tool = Tool(
            name="human_feedback",
            func=lambda filePath: self.human_feedback(filePath),
            description=(
                "Get Feeback from human via console and path and test again based on feedback provided."
            )
        )

        tools = [write_unit_test_tool, run_unit_test_tool, patch_code_from_test_tool, human_feedback_tool]

        return initialize_agent(
            llm=self.client,
            tools=tools,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=20,
            verbose=True
        )

    def _call_llm(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response.content.strip()
    
    def _get_tasks(self):
        # Define tasks: (filename, high-level instruction, needed config keys)
        planner_prompt = f"""
            You are a Python game architect and Pygame specialist.  
            Here’s my game manifest (JSON):
            ```json
            {self.manifest}

            RULES:
            - Each key in the manifest represents a config module that must map to one or more Python files.
            - Make sure no such files are present which has no config module.
            - If a value is an object (e.g. "screens", "objects"), generate one module per subkey (e.g. win_screen, apple).
            - **Important** Make sure tasks are in topological sorted order and there should be no cyclic dependencies.
            - Make sure to wirte individual module first and keep it follow single responsibility principle as much possible.
            - Create a central 'main.py' class that integrates and manages all modules in the correct order.
            
            - OUTPUT FORMAT (STRICT!):
            - Enclose your entire plan in triple backticks (```).
            - One line per file, exactly:
            filename.py | Detailed responsibility of this file including key classes, functions, APIs, and interaction notes | config filename | list of dependencies d1,d2..
            End with a line that reads:
            END_OF_PLAN
            Do not include any extra text, explanations or markdown beyond the back‐ticked block.
            """
        plan = self._call_llm(planner_prompt)
        tasks = self._parse_tasks(plan.split("END_OF_PLAN")[0])
        tasks = self._reorder_tasks(tasks)
        
        for task in tasks:    
            self._write_file(self.tasks_data, str(task), False)
            print(task)

        if self._ask_yes_no("Do you want to continue?"):
            return tasks
        
        return None
    
    # Precompute all valid config paths:
    def _list_paths(self, d, prefix=""):
        paths = []
        if isinstance(d, dict):
            for k, v in d.items():
                p = f"{prefix}.{k}" if prefix else k
                paths.append(p)
                paths += self._list_paths(v, p)
        return paths
    
    def _reorder_tasks(self, tasks: List[Task]) -> List[Task]:
        filename_set = {fn for fn, *_ in tasks}
        lookup: dict[str, Task] = { task[0]: task for task in tasks }

        graph: dict[str, list[str]] = defaultdict(list)
        indegree: dict[str, int] = { fn: 0 for fn in filename_set }

        for fn, _, _, deps in tasks:
            for dep in deps:
                dep_file = dep if dep in filename_set else f"{dep}.py"
                if dep_file in filename_set:
                    graph[dep_file].append(fn)
                    indegree[fn] += 1

        # Kahn’s algorithm
        q = deque([fn for fn, deg in indegree.items() if deg == 0])
        ordered: list[Task] = []

        while q:
            cur = q.popleft()
            ordered.append(lookup[cur])
            for nbr in graph[cur]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    q.append(nbr)

        if len(ordered) != len(tasks):
            print("⚠️ Warning: Some tasks couldn’t be fully ordered (cycle or missing dep).")
            # Optionally append the rest in any order:
            unordered = [t for t in tasks if t[0] not in {o[0] for o in ordered}]
            ordered.extend(unordered)

        return ordered

    def _parse_tasks(self, text: str) -> Task:
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
            if len(parts) != 4:
                raise ValueError(f"Line does not have exactly 4 parts: {line!r}")
            
            name, instruction, keys_str, deps_str = parts
            
            # Safely parse the keys list literal
            try:
                keys = [k.strip() for k in keys_str.split(",") if k.strip()]
                deps = [d.strip() for d in deps_str.split(",") if d.strip()]
            except Exception as e:
                raise ValueError(f"Failed to parse keys on line: {line!r}\n  {e}")
            
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise ValueError(f"Parsed keys is not a list of strings: {keys!r}")

            if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                raise ValueError(f"Parsed keys is not a list of strings: {deps!r}")

            if ".py" not in name:
                name = f"{name}.py"

            tasks.append((name, instruction, keys, deps))
        
        return tasks
    
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

    def _write_file(self, path: str, data: str, override: bool = True):
        mode = "w" if override else "a"
        with open(path, mode, encoding="utf-8") as f:
            f.write(f"{data}\n")
        print(f"✅ Wrote to {path} (override={override})")


    def _read_file(self, path: str):
        with open(path, "r") as file:
            content = file.read()
            return content
        
    def _clean_keys(self, raw_keys: List[Any]) -> List[str]:
        """
        Turn items like "'screens.win_screen.message_text'" or ast.Attribute nodes
        into clean strings: "screens.win_screen.message_text".
        """
        cleaned: List[str] = []
        for item in raw_keys:
            s = str(item)              # e.g. "'screens.win_screen.message_text'"
            s = s.strip().strip("'\"") # => screens.win_screen.message_text
            # keep only letters, digits, dots, and underscores
            s = re.sub(r"[^0-9A-Za-z._]", "", s)
            if s:
                cleaned.append(s)
        return cleaned
    
    import re

    import re

    def _get_by_path(self, config, path):
        current = config
        parts = path.split('.')
        for part in parts:
            # Parse key and optional index, e.g. objects[0]
            match = re.match(r"(\w+)(?:\[(\d+)\])?", part)
            if not match:
                # Unexpected format
                return None
            
            key = match.group(1)
            index = match.group(2)

            # Access dictionary key
            if isinstance(current, dict):
                current = current.get(key)
            else:
                # Can't get key from non-dict
                return None
            
            if current is None:
                return None

            # If there is an index, access list element
            if index is not None:
                idx = int(index)
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None

        return current

    def _write_code_and_tests(self, tasks):
    
        for filename, instruction, keys, deps in tasks:
            subconfig = []
            for key in keys:
                path = os.path.join(Constants.CONFIG_BASE, key)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        subconfig.append(json.load(f))

            print(f"Subconfig for {filename}:", subconfig)


            # Promt llm to generate class signature for given instruction and subconfig and capture results.
            stub_prompt = f"""
                You are a senior Python game developer creating modular Pygame code.

                ### TASK:
                Generate a stub implementation for the file: **`{filename}`**

                - **description**: {instruction}
                - **Relevant Data to build the game logic**:
                {json.dumps(subconfig, indent=2)}
                - **Depends On Classes**: {deps}

                ### OUTPUT FORMAT (STRICTLY THIS):
                - All necessary `import` statements for built-in modules and dependent classes.
                - Module-level docstring explaining the description of this file.
                - Class and/or function definitions only (no logic implemented).
                - Use `# TODO:` comments in method/function bodies.
                - Copy the game data as class-level constants (e.g. `SCREEN_WIDTH = 800`) and use them to develop logic.

                ### IMPORTANT:
                - Be minimal but accurate: focus on structure, interface, and reusability.
                - Your goal is to generate a working scaffold that will later be implemented.
                - Output only valid Python code — no explanations, comments, or markdown.

            """

            stub_code = self._call_llm(stub_prompt)
            print(f"STUB Code generated. {stub_code}")

            # Prompt llm to write implementation.
            impl_prompt = f"""
                You are an expert Pygame developer.

                File: `{filename}`  
                Instruction: {instruction}

                Stub:
                ```python
                {stub_code}

                - Replace all # TODO blocks with working code.
                - Use and import any required classes from the instruction.
                - Remove all docstrings.
                - Preserve the structure — do not rename anything.
                - Output the complete Python file only, no extra text or formatting.
            """
            impl_code = self._call_llm(impl_prompt)
            print(f"IMPL Code generated. {impl_code}")

            lint_prompt = f"""
                You are a Python linter.

                Task: Fix all linting and syntax issues in `{filename}` using `flake8` standards.

                Instructions:
                - Do not change logic — only fix style and syntax.
                - Remove all docstrings.
                - Return the full corrected Python file only, with no extra commentary.

                Code:
                    ```python
                {impl_code}
            """
            impl_code = self._call_llm(lint_prompt)
            print(f"LINT Code generated. {impl_code}")

            impl_code = self._strip_fences(impl_code)
            full_path = os.path.join(self.output_dir, filename)
            self._write_file(full_path, impl_code, True)

            self._run_unit_testing_agent(filename, impl_code)

    def _run_unit_testing_agent(self, filename: str, final_code: str) -> str:
        """
        Runs a coding agent to validate a module by:
        - Writing a unit test
        - Running the test
        - Patching code if the test fails

        Args:
            filename (str): Name of the file to test (e.g., "apple.py")
            final_code (str): Full source code of the file to be tested

        Returns:
            str: "ok" if the test passed or patch applied, or error stack if unresolved
        """

        prompt = f"""
            You are a smart Python unit testing agent with code understanding and patching skills.
            You're given the source file name and its full code.

            Your mission:
            1. Use the `write_unit_test` tool to generate a test class.
            2. Use the `run_unit_test` tool to execute the generated test class.
            3. If the test fails, use the `patch_code_from_test` tool to fix the original code.
            4. If the test class fails to compile, use the `patch_code_from_test` tool to fix the test class code.
            5. If patch is unable to resolve the issue after several tries, use the `human_feedback_tool` to get feeedback from human.
            6. Re-run the unit test until it passes or you determine it's unfixable.

            RULES:
            - Always use the tools. Do not invent or simulate test or patching code.
            - Only return "ok" if the final unit test passes.

            Input:
            ```json
            {{
            "filename": "{filename}",
            "code": "{final_code}"
            }}
            """

        result = self.agent.run(prompt)
        return result
    
    def _write_unit_test(self, class_filename: str) -> str:
        """
        Generates a unit test file named `Test_ClassName.py` for the specified class file.
        Assumes the class name matches the filename (e.g., Apple.py -> class Apple).
        """
        # ... logic here ...
        file_content = self._read_file(os.path.join(self.output_dir, class_filename))
        code, config, description = self.extract_parts(file_content)

        prompt = textwrap.dedent(f"""
            Write a unit test using **pytest**. The output must be valid Python code.
                                 
            ## Use relative import before importing file, this should be always be valid for all test files.
            ```
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            ```
            then import filename, filename = {class_filename}
            example:                                     
            from game_info(filename) import GameInfo, SCREEN_WIDTH, SCREEN_HEIGHT ... and so on.

            ### 🎯 Requirements:
            - Import the class/module correctly.
            - Use **pytest** best practices (e.g., fixtures, clear test function names).
            - Test only the **public API** exposed in the description below.

            ### 📦 Code: This is the implementation that should be tested using the config and description below.
            ```python
            {code}
            🛠 Config data : This config data is used to write the code implemented above.
            {config}
            📚 Description: This describes the purpose of the module and the public API that must be tested.
            {description}

            ✅ Generate a pytest unit test file that fully tests the functionality and behavior described above using the provided config and class code.
            """)
        
        raw_resposne = self._call_llm(prompt)
        test_code = self._strip_fences(raw_resposne)
        test_filename = f"Test_{class_filename.replace('.py', '')}.py"
        self._write_file(os.path.join(self.test_dir, test_filename), test_code)

        print(f"[✅] Unit test for '{class_filename}' generated successfully.")
        return test_filename
    
    def extract_parts(self, file_content):
        lines = file_content.splitlines()
        code_lines = []
        config_lines = []
        description_lines = []
        in_config = False
        for line in lines:
            # Detect start of config block
            if re.match(r"^\s*Config\s*=[\s\S]*", line) or re.match(r"^# DO NOT REMOVE", line):
                in_config = True
            if in_config:
                config_lines.append(line)
            else:
                code_lines.append(line)
            # Detect description line
            if line.strip().startswith("# Description"):  # capture description
                # assume rest of line is description
                desc = line.split("=", 1)[-1].strip()
                description_lines.append(desc)
        # Clean code: remove trailing empty lines
        while code_lines and code_lines[-1].strip() == "":
            code_lines.pop()
        code = "\n".join(code_lines).strip()
        config = "\n".join(config_lines).strip()
        description = " ".join(description_lines).strip()
        return code, config, description
    
    def _run_unit_test(self, filename: str) -> dict:
        """
        Runs the corresponding unit test file `Test_filename.py` and returns the output,
        including stdout, stderr, and return code.
        """
        # ... logic here ...
        test_path = os.path.join(self.test_dir, filename)

        try:
            result = subprocess.run(
                ["pytest", "-s", "-v", test_path, "--tb=short"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                output["status"] = "success"
                output["traceback"] = None
            else:
                output["status"] = "error"
                output["traceback"] = result.stderr

            print(f"[✅] Unit test for '{filename}' executed successfully.")
            return output

        except FileNotFoundError:
            return {
                "status": "error",
                "stdout": "",
                "stderr": "pytest not found. Please ensure pytest is installed.",
                "returncode": -1,
                "traceback": "pytest executable not found in system PATH."
            }

    def _patch_code_from_test(self, args) -> str:
        try:
            # Split the input string
            if "||" not in args:
                return "Invalid input format. Expected 'filename||test_output'."

            filename, test_output = args.split("||", 1)
            if "Test" in filename:
                filepath = os.path.join(self.test_dir, filename)
            else:
                filepath = os.path.join(self.output_dir, filename)

            # Read original file content
            code = self._read_file(filepath)

            patch_prompt = f"""
                You are an expert software engineer.

                Your task is to patch the following Python source code based on test errors. 
                You will receive:
                1. The source code that failed.
                2. The pytest output that shows the errors.

                Analyze the test output and modify the code to:
                - Add any missing method or property stubs.
                - Make minimal changes to fix the issue.
                - Do NOT remove existing functionality unless clearly broken.
                - Use clear `# TODO` comments for incomplete method implementations.

                Return the complete corrected code (as a single code block, no explanation).

                --- FILE START ---
                {code}
                --- FILE END ---

                --- TEST OUTPUT START ---
                {test_output}
                --- TEST OUTPUT END ---

                Now return the updated code below:
                ```python
                """

            raw_patch = self._call_llm(patch_prompt)
            fix_code = self._strip_fences(raw_patch)
            self._write_file(filepath, fix_code)

            return f"[✅] Code in '{filename}' patched based on test results."

        except Exception as e:
            return f"[❌] Exception occurred while patching: {str(e)}"
        
    def human_feedback(self, filepath):
        
        try:
            user_input = inputimeout(prompt=f'Enter Feedback for {filepath}: ', timeout=100)
        except TimeoutOccurred:
            user_input = 'Ok'
        return user_input


# Runner
if __name__ == "__main__":
    agent = DeveloperAgent()
    agent.develop_game()

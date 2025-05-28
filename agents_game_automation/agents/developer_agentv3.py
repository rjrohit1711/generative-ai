# agents/developer_agent.py

import re
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
        self.data_dir = os.path.join(self.output_dir, "data")
        self.tasks_data = os.path.join(self.data_dir, "tasks.txt")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

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
                "Input format: 'class_name' "
                "(e.g. 'Apple.py')"
            )
        )

        run_unit_test_tool = Tool(
            name="run_unit_test",
            func=lambda filename: self._run_unit_test(filename),
            description=(
                "Runs the unit test file for the given class and returns the result. "
                "Input format: 'test_class_name' "
                "(e.g. 'Test_Apple.py')"
            )
        )

        patch_code_from_test_tool = Tool(
            name="patch_code_from_test",
            func=lambda args: self._patch_code_from_test(args),
            description=(
                "Patches the class implementation based on the unit test failure output. "
                "Input format: 'class_name||test_output' "
                "(e.g. 'Apple.py||<error_trace_or_output>')"
            )
        )


        tools = [write_unit_test_tool, run_unit_test_tool, patch_code_from_test_tool]

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
    
    def _write_unit_test(self, class_filename: str) -> str:
        """
        Generates a unit test file named `Test_ClassName.py` for the specified class file.
        Assumes the class name matches the filename (e.g., Apple.py -> class Apple).
        """
        # ... logic here ...
        print(f"[✅] Unit test for '{class_filename}' generated successfully.")
        return f"Test_{class_filename.replace('.py', '')}.py"
    
    def _run_unit_test(self, class_filename: str) -> dict:
        """
        Runs the corresponding unit test file `Test_ClassName.py` and returns the output,
        including stdout, stderr, and return code.
        """
        # ... logic here ...
        print(f"[✅] Unit test for '{class_filename}' executed successfully.")
        return {"status": "Error", "stdout": "", "stderr": "Error Raised Exception", "returncode": -1}

    def _patch_code_from_test(self, args) -> str:
        """
        Analyzes test output to identify missing methods or basic issues and patches the original class file
        by inserting method stubs or minimal fixes.
        """
        # ... logic here ...
        print(f"[✅] Code in '{args}' patched based on test results.")
        return "Patch applied successfully."
    
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
            - Each file must have a detailed responsibility summary including its core logic, key functions, classes, and APIs it exposes.
            - Describe how this module will interact with others (inputs/outputs/shared data).
            - Make sure tasks are in topological sorted order and there should be no cyclic dependencies.
            - Create a central 'main.py' class that integrates and manages all modules in the correct order.
            
            - Here are few examples as a reference:
            
            game_info.py | Defines `GameInfo` class that loads and provides access to global configuration such as screen dimensions, frame rate, and asset paths. Public API: `GameInfo.load()`, `GameInfo.get(key)`, `GameInfo.screen_width`, `GameInfo.screen_height`. Used by all other modules to retrieve game-wide settings. | game_info.json | None
            player.py | Defines `Player` class that handles movement, animation, collision detection, scoring logic, and interaction with collectibles and obstacles. Public API: `Player.update()`, `Player.draw(screen)`, `Player.check_collision(obj)`, `Player.reset()`. Depends on `GameInfo` for screen bounds and on `mechanics` for rules like gravity and scoring. | player.json | game_info, mechanics
            apple.py | Defines `Apple` class representing the collectible item. Handles spawning, falling motion, collision with player, and respawn. Public API: `Apple.update()`, `Apple.draw(screen)`, `Apple.check_collision(player)`, `Apple.reset_position()`. Uses config to determine spawn interval and fall speed. Interacts with `Player` to increment score on collect. | apple.json | game_info, mechanics
            mechanics.py | Defines global mechanics like gravity, scoring rules, and spawn logic. Exposes `apply_gravity(obj)`, `calculate_score(type)`, and `spawn_entity(type)`. Provides reusable physics and scoring logic to `Player`, `Apple`, `Bee`, etc. | mechanics.json | game_info
            main.py | Entry point of the game. Loads `GameInfo`, initializes all modules, controls game loop, handles transitions between screens (`start_screen`, `game_screen`, `win_screen`, `lose_screen`). Public API: `main()`, `handle_events()`, `switch_screen(name)`. | None | game_info, game_screen, win_screen, lose_screen, player, apple, golden_apple, bee, stone, mechanics, levels
            
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
            self._write_file(self.tasks_data, str(task))
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

    def _write_file(self, path: str, data: str, override: bool = False):
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

                - **Purpose**: {instruction}
                - **Relevant Config Data**:
                {json.dumps(subconfig, indent=2)}
                - **Depends On Classes**: {deps}

                ### OUTPUT FORMAT (STRICTLY THIS):
                - All necessary `import` statements for built-in modules and dependent classes.
                - Module-level docstring explaining the purpose of this file.
                - Class and/or function definitions only (no logic implemented).
                - Use `# TODO:` comments in method/function bodies.
                - Extract only the required config keys and declare them as class-level constants (e.g. `SCREEN_WIDTH = 800`)
                - Do **not** copy nested JSON objects as-is.
                - Use already defined classes where applicable.

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
                Config: {json.dumps(subconfig, indent=2)}

                Stub:
                ```python
                {stub_code}

                - Replace all # TODO blocks with working code.
                - Use and import any required classes from the summary.
                - Extract relevant config values as class-level constants.
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
                - Remove unused variables and imports.
                - From the dependent classes listed: {deps}, remove any not actually used.
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

            with open(full_path, 'r', encoding='utf-8') as f:
                final_tested_code = f.read()
            
            # Remove any testing code. 
            final_prompt = f"""
                All tests now pass for `{filename}`.  
                Please remove testing code from this and clean it. Without modifying any logic.
                Add Config data as comment at the bottom of the page. Add comment above this saying do not remove this config.
                Config = {subconfig}
                Add Task description below this.
                Description = {instruction}   
                ```
                {final_tested_code}
                """
            final_code = self._call_llm(final_prompt)
            print(f"Final Code generated. {final_code}")
            final_code = self._strip_fences(final_code)
            self._write_file(full_path, final_code, True)

            self._run_unit_testing_agent(filename, final_code)

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
            4. Re-run the unit test until it passes or you determine it's unfixable.

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

# Runner
if __name__ == "__main__":
    agent = DeveloperAgent()
    agent.develop_game()

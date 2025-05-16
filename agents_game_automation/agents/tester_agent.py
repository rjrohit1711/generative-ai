# agents/tester_agent.py

import subprocess
import py_compile
import os
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

class TesterAgent:
    def __init__(
        self,
        model: str = "mistralai/mistral-small-24b-instruct"
    ):
        load_dotenv()
        api_key = os.getenv("MISTRALAI_API_KEY")
        if not api_key:
            raise ValueError("MISTRALAI_API_KEY not set in .env")

        client = ChatOpenAI(
            model_name = model,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            openai_api_key=api_key,
            temperature = 0.2
        )
        
        memory =  ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.output_dir = "bin/source/game/test"
        os.makedirs(self.output_dir, exist_ok=True)
    
        test_code_tool = Tool(
            name="test_code",
            func=lambda filePath: self.test_code(filePath),
            description="Run the current file to see if it complies and runs properly or else return the error stack."
        )

        patch_code_tool = Tool(
            name="patch_code",
            func=lambda args: self._patch_code_tool(args),
            description=(
                "Apply an AI-generated patch to a file based on an error. "
                "Input format: 'relative/path/to/file.py||<error message>||<AI Thought>'."
                "Or if there is a human feedback."
                "Input format: 'relative/path/to/file.py||<error message>||<Human feedback>'."
            )
        )

        human_feedback_tool = Tool(
            name="human_feedback",
            func=lambda filePath: self.human_feedback(filePath),
            description=(
                "Get Feeback from human via console and path and test again based on feedback provided."
            )
        )

        tools = [test_code_tool, patch_code_tool, human_feedback_tool]

        self.agent = initialize_agent(
            llm=client,
            tools=tools,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,# ← enable automatic retry on parse errors
            memory=memory,
            verbose=True
        )

    def _llm(self, prompt: str) -> str:

        load_dotenv()
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY not set in .env")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model = "qwen/qwen2.5-coder-32b-instruct"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert code tester."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    
    def test_agent(self, path, summary = None, subconfig = None):
        directory = os.path.dirname(path)
        files_list =  [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        print(f"Files: {files_list}")
        prompt = f"""
            You are an autonomous coding agent.

            You can call tools using this format:

            Thought: I need to test the code.
            Action: test_code
            Action Input: bin/source/gamev3/Player.py

            Thought: The test failed with an error in Enemy.py, so I’ll patch it.
            Action: patch_code_tool
            Action Input: bin/source/gamev3/Enemy.py||NameError: name 'pygame' is not defined||Add `import pygame` at the top.

            Thought: Tests passed. I will now ask for feedback.
            Action: human_feedback
            Action Input: All tests passed. Ask user if anything else should be improved.

            ---

            Your steps:
            1. Run `test_code({path})`.
            2. If there is an error, extract the real file from the traceback.
            3. Call `patch_code_tool(actual_file, <error>, <AI Thought>)`.
            4. Once `test_code` succeeds, ask for `human_feedback` and decide if another patch is needed.
            5. Repeat until the code works (or human_feedback is positive). Every 5 iterations, always call `human_feedback`.
            6. You may use files from: {files_list}
            7. Summary of project: {summary}
            8. Subconfig used for this file: {subconfig}

            Begin.
            """
        self.agent.run(prompt)

    def test_code(self, path: str) -> str:
        """
        1) Compile the file to catch syntax errors.
        2) Run it as a standalone script.
        Returns "OK" if both compile and run succeed, otherwise returns the error output.
        """
        # 1) Syntax check
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as compile_err:
            err = f"SyntaxError during compile:\n{compile_err}"
            print("❌", err)
            return err

        # 2) Run the script
        print(f"🧪 Running {os.path.basename(path)} …")
        proc = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True
        )

        if proc.returncode == 0:
            print("✅ Script ran successfully!")
            return "OK"
        else:
            error_output = proc.stdout + proc.stderr
            print("❌ Runtime errors:\n", error_output)
            return error_output
    
    def _patch_code_tool(self, payload: str) -> str:
        """
        Parses a single string payload of the form:
            'file_path||error message||thought process'
        then calls the two-arg patch_code method.
        """
        try:
            print("Extracting patch info...")
            file_path, error, thoughts = payload.split("||", 2)
        except ValueError:
            return "Invalid input. Use 'file_path||error message||AI thought'."
        return self.patch_code(file_path.strip(), error.strip(), thoughts.strip())

        
    def patch_code(self,filePath, error, thoughts):
        print("Executing patch...")

        """
        Prompts the LLM to patch the given file based on the error message.
        Returns the patched source code as a string.
        """
        # 1. Read the current code
        with open(filePath, "r", encoding="utf-8") as f:
            original = f.read()

        # 2. Build a focused prompt
        prompt = f"""
            I attempted to run `{filePath}` and encountered this error:
            {error}
            My thought process {thoughts}
            Here is the current contents of `{filePath}`:
            ```python
            {original}
            ```
            Please return the complete, corrected contents of {filePath} that resolve the above error.
            Do NOT include any explanations—ONLY the code.
            Enclose your response in triple backticks like python ... so it can be extracted cleanly.
            """

        patch_response = self._llm(prompt)
        patched_code = self._strip_fences(patch_response)
        self._write_file(filePath, patched_code)
        return "Patch done"
    
    def human_feedback(self, filepath):
        inputs = input(f"Enter feedback for file: {filepath}")
        return inputs

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


    def _write_file(self, filePath: str, code: str):
        with open(filePath, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Wrote {filePath}")

if __name__ == "__main__":

    tester_agent = TesterAgent()
    dir = "bin/source/gamev3/"
    file = "Game.py"
    path = os.path.join(dir, file)
    with open(path, "r", encoding="utf-8") as f:
            code = f.read()
    tester_agent.test_agent(path)

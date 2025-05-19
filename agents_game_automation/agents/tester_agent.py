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
from inputimeout import inputimeout, TimeoutOccurred

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
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=50, 
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
    
    def test_agent(self, path, summary = None, subconfig = None, classes = None):
        directory = os.path.dirname(path)
        files_list =  [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        
        print(f"Files: {files_list}")
        prompt = f"""
            You are an autonomous coding agent. Use available tools to debug and improve the code.

            You may use files from: {directory}/  
            Available files: {files_list}  
            Project Summary: {summary}  
            Config for Entire project it will have all assets path if required: {subconfig}

            Instructions:
            1. Run `test_code(<path>)`.
            2. If an error occurs, call `patch_code_tool(<file>||<error>||<fix>)`, then repeat step 1.
            3. If the test passes, call `human_feedback`.
            4. If the same error or patch repeats multiple times, assume you're stuck in a loop and request human feedback before continuing.

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
        
        try:
            user_input = inputimeout(prompt=f'Enter Feedback for {filepath}: ', timeout=100)
        except TimeoutOccurred:
            user_input = 'Ok'
        return user_input

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
    
    summary_path = os.path.join(dir, "Summary/summary.txt")
    with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
    json_path = "bin/source/game_configv3.json"
    with open(json_path, "r", encoding="utf-8") as f:
            json = f.read()
    
    tester_agent.test_agent(path, summary, json)

# agents/tester_agent.py

import subprocess
import os
from openai import OpenAI
from dotenv import load_dotenv

class TesterAgent:
    def __init__(
        self,
        model: str = "mistralai/mistral-small-24b-instruct"
    ):
        load_dotenv()
        api_key = os.getenv("MISTRALAI_API_KEY")
        if not api_key:
            raise ValueError("MISTRALAI_API_KEY not set in .env")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = model
        self.output_dir = "bin/source/game/test"
        os.makedirs(self.output_dir, exist_ok=True)

    def _llm(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert code tester."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    def test_code(self, filename, code):
        """
        Agent used to test code.
        """
        print(f"🧪 Testing {filename}")
        testing_prompt = f"""
        You are a test-writing assistant. Given the following Python code, write a `pytest`-based test file.

        Requirements:
        1. Just perform unit testing of the logic which any external import. 
        2. The test must return `0` if all tests pass.
        3. If any test fails, return a list of failed assertions and error messages.
        4. Generate valid pyhton code. 
        5. Enclose code in between ``` ``` quotes so i can extract it.

        Module Code:
        ```python
        {code}
        ```
        """
        testing_code = self._llm(testing_prompt)
        final_code = self._strip_fences(testing_code)
        self._write_file(f"test_{filename}", final_code)

        print("🧪 Running Test...")
        result = subprocess.run(["pytest", f"{self.output_dir}/test_{filename}"], capture_output=True, text=True)
        print("📄 Test Output:\n", result.stdout)
        return result.returncode


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
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Wrote {filename}")

if __name__ == "__main__":
    with open("bin/source/gamev2/play.py", "r", encoding="utf-8") as f:
        code = f.read()

    result = TesterAgent().test_code("play.py", code)
    print("🧪 Result:\n", result)

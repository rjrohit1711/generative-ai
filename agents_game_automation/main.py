# main.py

from core.orchestrator import Orchestrator

def main():
    orchestrator = Orchestrator()
    orchestrator.run(
        prompt="Create a panda racing game for mobile"
    )

if __name__ == "__main__":
    main()
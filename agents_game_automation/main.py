# main.py

from core.orchestrator import Orchestrator

def main():
    orchestrator = Orchestrator()
    orchestrator.run(
        prompt="Create a simple game in which Dragon flies and collect coins while avoiding walls they act as obstacles."
    )

if __name__ == "__main__":
    main()
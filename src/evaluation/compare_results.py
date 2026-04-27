import json
import os

RESULTS_DIR = "evaluation/results"


def load_results(file):
    with open(file, "r") as f:
        return json.load(f)


def main():
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]

    print("\n📊 COMPARISON:\n")

    for file in files:
        data = load_results(os.path.join(RESULTS_DIR, file))

        answered = sum(1 for d in data if "not found" not in d["answer"].lower())

        print(f"{file}: {answered}/{len(data)} answered")


if __name__ == "__main__":
    main()
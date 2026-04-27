import os

scripts = [
    "run_simple.py",
    "run_hyde.py",
    "run_contextual.py",
    "run_multi_query.py",
    "run_multimodal.py",
    "run_self.py",
    "run_memo.py",
]

for script in scripts:
    rag_type = script.replace("run_", "").replace(".py", "")
    result_file = f"final_evaluation/results/{rag_type}_results.json"

    print(f"\n🚀 Running {script}")
    os.system(f"python -m src.pipelines.{script.replace('.py','')}")
    os.system(f"python -m src.evaluation.evaluate_and_compare --file {result_file}")
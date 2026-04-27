import argparse
import shutil
from pathlib import Path

from src.core.config import EVAL_DIR

RESULTS_DIR = Path(EVAL_DIR) / "results"
SCORES_DIR = Path(EVAL_DIR) / "scores"


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Deleted directory: {path.as_posix()}")
    elif path.exists():
        path.unlink()
        print(f"Deleted file: {path.as_posix()}")


def clean_rag_type(rag_type: str) -> None:
    delete_path(RESULTS_DIR / f"{rag_type}_results.json")
    delete_path(SCORES_DIR / rag_type)


def clean_all() -> None:
    if RESULTS_DIR.exists():
        for file_path in RESULTS_DIR.glob("*_results.json"):
            delete_path(file_path)

    if SCORES_DIR.exists():
        for child in SCORES_DIR.iterdir():
            delete_path(child)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean evaluation outputs")
    parser.add_argument("--rag-type", type=str, default=None, help="Clean a single RAG type, e.g. hyde")
    parser.add_argument("--all", action="store_true", help="Clean all generated results and scores")
    args = parser.parse_args()

    if args.all:
        clean_all()
        return

    if args.rag_type:
        clean_rag_type(args.rag_type)
        return

    parser.error("Provide either --rag-type or --all")


if __name__ == "__main__":
    main()

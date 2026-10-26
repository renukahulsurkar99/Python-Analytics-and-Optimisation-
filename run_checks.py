"""Run the optimisation cases and locally supplied AED analysis scripts."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Reduce model tuning for functional checks.")
    parser.add_argument("--optimisation-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["AED_QUICK_TEST"] = "1" if args.quick else "0"
    environment.setdefault("LOKY_MAX_CPU_COUNT", "2")
    scripts = [f"optimisation/{name}" for name in [
        "task1_cost.py", "task2_minimax.py", "task2_range.py", "task3_skills.py"
    ]]
    if not args.optimisation_only:
        from project_config import DATA_PATH
        if not DATA_PATH.is_file():
            parser.error("The local AED dataset is missing. See data/README.md.")
        scripts.extend(f"machine_learning/{name}" for name in [
            "Task_4.py", "Task_5.py", "Logistic_Regression.py", "Decision_tree.py", "Random_Forest.py"
        ])
    logs = root / "local_outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in scripts:
        print(f"Running {name}", flush=True)
        with (logs / (Path(name).stem + ".log")).open("w") as log:
            result = subprocess.run([sys.executable, str(root / name)], cwd=root, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Failed: {name}. See {logs / (Path(name).stem + '.log')}")
    print("All requested scripts completed successfully.")


if __name__ == "__main__":
    main()

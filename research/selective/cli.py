import argparse
import json
from pathlib import Path
from .data import read_csv
from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Selective binary-classification experiment"
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--label", default="label")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--maximum-error", type=float, default=0.15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = run_experiment(
            read_csv(args.csv, args.label),
            seed=args.seed,
            maximum_error=args.maximum_error,
        )
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n")
        print(rendered)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

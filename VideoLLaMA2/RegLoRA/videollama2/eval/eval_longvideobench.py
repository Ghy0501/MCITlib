import json
import argparse
from tabulate import tabulate


def main():
    args = parse_args()

    # Load predictions (jsonl)
    with open(args.pred_path, 'r') as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    assert total > 0, "Empty prediction file."

    video_correct = 0
    subtitle_correct = 0

    for r in records:
        if r['pred'] == r['gt']:
            video_correct += 1
        if r.get('pred_sub', None) == r['gt']:
            subtitle_correct += 1

    video_acc = video_correct * 100.0 / total
    subtitle_acc = subtitle_correct * 100.0 / total
    gain = subtitle_acc - video_acc

    # Print summary
    print(f"Evaluation file: {args.pred_path}")
    print(f"Total samples: {total}\n")

    table = [
        ["Setting", "Accuracy (%)"],
        ["Video only", f"{video_acc:.2f}"],
        ["Video + Subtitle", f"{subtitle_acc:.2f}"],
        ["Subtitle Gain", f"{gain:+.2f}"],
    ]

    print(tabulate(table, headers="firstrow", tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LongVideoBench MCQA results"
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to LongVideoBench prediction jsonl file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

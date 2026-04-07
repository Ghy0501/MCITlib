import json
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate prediction scores.")
    parser.add_argument("input_path", type=str, help="Path to the input JSON file.")
    parser.add_argument("result_path", type=str, help="Path to save the evaluation results.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input_path
    result_path = args.result_path

    yes_count = 0
    no_count = 0
    total = 0
    total_score = 0

    # Read the input file and process each line
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            ans = item.get("answer", "").strip()
            pred = item.get("pred", "").strip()

            ans_key = ans.split(".")[0].strip()
            pred_key = pred.split(".")[0].strip()

            total += 1
            if ans_key == pred_key:
                yes_count += 1
                total_score += 5
            else:
                no_count += 1
                total_score += 0

    # Calculate accuracy and average score
    accuracy = yes_count / total if total > 0 else 0
    avg_score = total_score / total if total > 0 else 0

    # Print results
    print(f"Yes count: {yes_count}")
    print(f"No count: {no_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average score: {avg_score:.4f}")

    # Write results to the result_path
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Yes count: {yes_count}\n")
        f.write(f"No count: {no_count}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Average score: {avg_score:.4f}\n")


if __name__ == "__main__":
    main()
import os
import argparse
import json

# ===================== Configuration =====================
SCORE_CORRECT = 5  # Score for correct answers (0-5 scale)
SCORE_INCORRECT = 0  # Score for incorrect answers
# Strict match set for choice/judgment (lowercase only)
STRICT_MATCH_ANSWERS = {'a', 'b', 'c', 'd', 'e', 'yes', 'no'}
# ==========================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ultra Simple QA Evaluation (Rule-Based)")
    parser.add_argument("--pred_path", required=True, help="Path to prediction file (one JSON per line)")
    parser.add_argument("--output_json", required=True, help="Path to save full evaluation JSON")
    parser.add_argument("--result_path", required=True, help="Path to save evaluation metrics TXT")
    return parser.parse_args()

def clean_answer(text):
    """Clean answer: convert to string -> strip spaces -> lowercase"""
    if not isinstance(text, str):
        text = str(text)
    return text.strip().lower()

def evaluate_single(gt_ans, pred_ans):
    """
    Evaluate single QA sample with rule-based logic
    :param gt_ans: Ground truth answer
    :param pred_ans: Predicted answer
    :return: (eval_result: str, score: int) -> ("yes"/"no", 0/5)
    """
    gt = clean_answer(gt_ans)
    pred = clean_answer(pred_ans)
    # Strict match (==) for choice/judgment, contain match (in) for short answer
    is_correct = pred == gt if gt in STRICT_MATCH_ANSWERS else pred in gt or gt in pred
    return "yes" if is_correct else "no", SCORE_CORRECT if is_correct else SCORE_INCORRECT

def main():
    args = parse_arguments()
    # Check if input file exists
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_path}")

    # Initialize statistics and result storage
    total_samples = 0
    correct_count = 0
    incorrect_count = 0
    total_score = 0
    full_results = {}

    # Process data line by line
    print(f"Processing data from: {args.pred_path}")
    with open(args.pred_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            total_samples += 1
            # Skip empty lines
            if not line:
                print(f"Skip empty line {total_samples}")
                continue

            try:
                # Parse sample and extract key info
                sample = json.loads(line)
                sample_id = str(sample['id'])
                question = sample['question']
                gt_answer = sample['answer']
                pred_answer = sample['pred']

                # Evaluate and store result
                eval_res, score = evaluate_single(gt_answer, pred_answer)
                full_results[sample_id] = {
                    "original": {"id": sample_id, "question": question, "ground_truth": gt_answer, "prediction": pred_answer},
                    "evaluation": {"result": eval_res, "score": score}
                }

                # Update statistics
                total_score += score
                if eval_res == "yes":
                    correct_count += 1
                else:
                    incorrect_count += 1

            except json.JSONDecodeError:
                print(f"Line {total_samples}: Invalid JSON format")
            except KeyError as e:
                print(f"Line {total_samples}: Missing required key - {e}")
            except Exception as e:
                print(f"Line {total_samples}: Processing error - {str(e)[:50]}")

    # Calculate final metrics
    accuracy = 0.0
    avg_score = 0.0
    evaluated_count = correct_count + incorrect_count
    if evaluated_count > 0:
        accuracy = correct_count / evaluated_count
        avg_score = total_score / evaluated_count

    # Save full evaluation results
    with open(args.output_json, 'w', encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to: {args.output_json}")

    # Save metrics to TXT
    with open(args.result_path, 'w', encoding="utf-8") as f:
        f.write("=== QA Evaluation Metrics (Rule-Based) ===\n")
        f.write(f"Total input samples: {total_samples}\n")
        f.write(f"Successfully evaluated samples: {evaluated_count}\n")
        f.write(f"Correct answers: {correct_count}\n")
        f.write(f"Incorrect answers: {incorrect_count}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Average score (0-5): {avg_score:.4f}\n")

    # Print final metrics to console
    print("\n=== Final Evaluation Metrics ===")
    print(f"Total Input Samples:    {total_samples}")
    print(f"Successfully Evaluated: {evaluated_count}")
    print(f"Correct Answers:        {correct_count}")
    print(f"Incorrect Answers:      {incorrect_count}")
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Average Score (0-5):    {avg_score:.4f}")
    print("========================================")
    print(f"Metrics saved to: {args.result_path}")

if __name__ == "__main__":
    main()
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Select the routed answer by question ID in a single CL-VISTA cell."""
import argparse
import json
from pathlib import Path

from .protocol import ROUTER_LABELS, routed_predictions


def read_predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--experts", type=json.loads, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if set(args.experts) - set(ROUTER_LABELS.values()):
        parser.error("expert names must correspond to the historical CL-VISTA label pool")
    questions = json.loads(args.questions.read_text())
    router = read_predictions(args.router_predictions)
    experts = {expert: read_predictions(path) for expert, path in args.experts.items()}
    routed, audit = routed_predictions(questions, router, experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_name("route-coverage.json").open("x") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    with args.output.open("x") as handle:
        for row in routed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if audit["missing"]:
        raise ValueError(f"Selected experts produced {audit['selected']} of {audit['expected']} requested answers")


if __name__ == "__main__":
    main()

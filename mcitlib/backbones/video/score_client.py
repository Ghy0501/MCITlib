# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Request scoring of a generated CL-VISTA cell from the local native judge."""
import argparse
import json
import os
import socket
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judge_directory", type=Path, help="Runtime control directory containing judge.sock")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    predictions = args.predictions.resolve(strict=True)
    os.chdir(args.judge_directory)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(7200)
        client.connect("judge.sock")
        client.sendall((json.dumps({"predictions": str(predictions)}) + "\n").encode())
        with client.makefile("rb") as handle:
            response = json.loads(handle.readline())
    print(json.dumps(response, indent=2))
    return int(response["status"] != "scored")


if __name__ == "__main__":
    raise SystemExit(main())

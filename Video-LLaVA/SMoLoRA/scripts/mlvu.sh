#!/bin/bash

# 设置路径
SCRIPT_DIR="/your_path/MCITlib_v3/Video-LLaVA/SMoLoRA/scripts"

# 启动 1.py
echo "Starting 1.py..."
python3 $SCRIPT_DIR/1.py
if [ $? -ne 0 ]; then
    echo "Error occurred while running 1.py. Exiting..."
    exit 1
fi

# 启动 2.py
echo "Starting 2.py..."
python3 $SCRIPT_DIR/2.py
if [ $? -ne 0 ]; then
    echo "Error occurred while running 2.py. Exiting..."
    exit 1
fi

# 启动 3.py
echo "Starting 3.py..."
python3 $SCRIPT_DIR/3.py
if [ $? -ne 0 ]; then
    echo "Error occurred while running 3.py. Exiting..."
    exit 1
fi

echo "All scripts have completed successfully."

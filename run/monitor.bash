#!/bin/bash

TARGET_PROCESS="my_target_app"
PYTHON_SCRIPT="python iii.py"
CHECK_INTERVAL=5
# -------------------------

echo "Monitoring process '$TARGET_PROCESS'..."
echo "On exit, run '$PYTHON_SCRIPT'"

while pgrep -x "$TARGET_PROCESS" > /dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process '$TARGET_PROCESS' exited."
echo "Starting script..."

$PYTHON_SCRIPT

echo "Script finished."


while ps -p 145473 > /dev/null; do sleep 5; done && python iii.py
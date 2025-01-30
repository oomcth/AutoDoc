#!/bin/bash
# stop_recording.sh

if [ -f recording_pid.txt ]; then
    pid=$(cat recording_pid.txt)
    kill $pid 2>/dev/null
    rm recording_pid.txt
    echo "Enregistrement arrêté."
else
    echo "Aucun enregistrement en cours."
fi
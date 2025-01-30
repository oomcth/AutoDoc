#!/bin/bash
# start_recording.sh

rec output.wav &
echo $! > recording_pid.txt
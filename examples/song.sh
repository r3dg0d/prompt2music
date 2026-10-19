#!/usr/bin/env bash
# Vocal song from structured lyrics (needs a vocal model: ace-step-1.5, heartmula-3b, yue2-3b).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

prompt2music lyrics validate "$HERE/lyrics.txt" --model ace-step-1.5
prompt2music song --lyrics "$HERE/lyrics.txt" --genre "alternative rock" \
  --mood "bittersweet" --vocal-style "warm male vocal" --model ace-step-1.5 \
  --duration 150 --output song.wav

# Extend it and score a video with the result
prompt2music continue --input song.wav --duration 30 --mode outro \
  --model stable-audio-3-medium -o song-extended.wav

#!/usr/bin/env bash
# Instrumental examples. Pick any installed model; `--model auto` chooses one.
set -euo pipefail

prompt2music generate \
  --prompt "Dark cyberpunk synthwave with heavy bass, atmospheric pads, and futuristic percussion" \
  --duration 120 --instrumental --model "${MODEL:-auto}" --output cyberpunk.wav

prompt2music generate "Epic orchestral boss battle music with aggressive strings and cinematic percussion"

prompt2music generate "Aggressive industrial techno, distorted kick drums, dark atmosphere" \
  --bpm 145 --key Fm --format flac

# Queue in the background and come back later
JOB=$(prompt2music --json generate "Dreamy ambient electronic music with slow evolving textures" \
  --duration 180 --background | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
prompt2music jobs wait "$JOB"

#!/bin/bash
# submit.sh — usage: ./submit.sh <sampling> <reward>
SAMPLING="${1:?Missing sampling}"
REWARD="${2:?Missing reward}"
JOB_NAME="grpo_${SAMPLING}_${REWARD}_a100"

sbatch --job-name="$JOB_NAME" \
	--output="${JOB_NAME}-%j.out" \
	--error="${JOB_NAME}-%j.err" \
	run_p100_1gpus.sh "$SAMPLING" "$REWARD"

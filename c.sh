#!/bin/sh -e

# export TORCH_FORCE_WEIGHTS_ONLY_LOAD=true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python DQN_check_solution.py

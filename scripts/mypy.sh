#!/usr/bin/env bash
# Run type checking over the python code.
set -e
echo 'Starting mypy checks'
mypy skip_phrase --ignore-missing-imports
echo -e "mypy checks passed\n"
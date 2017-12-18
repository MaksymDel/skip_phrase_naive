#!/usr/bin/env bash
# Run our linter over the python code.

set -e
echo 'Starting pylint checks'
pylint -d locally-disabled,locally-enabled -f colorized skip_gram_phrase tests
echo -e "pylint checks passed\n"
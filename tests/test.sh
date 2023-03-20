#!/bin/bash
#
# General purpose testing script.
# Meant to be called from project root after checkout, will create a temp venv
# an run tests.

set -eu

DIR=$(mktemp -d)
echo "::: Creating environment"
python -m venv "$DIR/venv"
. "$DIR/venv/bin/activate"
pip --no-cache-dir install wheel
echo "::: Installing dependencies"
pip --no-cache-dir install ".[test]" ".gemz-galp[test]"
echo "::: Running tests"
pytest "--ignore=$DIR" -k 'not peer and not nonlinear_shrinkage' -v

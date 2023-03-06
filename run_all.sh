#!/bin/bash
set -e

for mdfile in *.md; do
	echo "Running $mdfile..."
	PYTHONPATH=. python -X dev <(python -m jupytext "$mdfile" --to py -o -)
done

#!/bin/bash
#
# General purpose testing script.
# Can be used to set up various test environment and run parts of the test suite

set -eu

case "$1" in
	singularity)
		# Run isolated tests in a singularity container
		;;
	tmp-clone)
		# Clone the repository in a new temp dir then run tests on it
		DIR=$(mktemp -d)
		echo "Running tests on fresh clone in $DIR"
		cd $DIR
		git clone --recurse-submodules=yes git@github.com:emorice/gemz
		cd gemz
		tests/test.sh local-venv
		;;
	local-venv)
		# Set up a python virtual venv, then run the tests in them
		python -m venv venv
		. venv/bin/activate
		pip install .[test] ./gemz-galp[test]
		"$0" local
		;;
	local)
		# Actually run the tests
		pytest -k svd
		;;
	*)
		echo "Unknown, or no subcommand given"
esac

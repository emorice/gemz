#!/bin/bash
#
# General purpose testing script.
# Can be used to set up various test environment and run parts of the test suite

set -eu

case "$1" in
	singularity)
		# Run isolated tests in a singularity container
		# The working directory, which must be the project root, is
		# bound inside the container, and this script called recursively
		# inside
		# As a result, the project root is used as the working dir for
		# the tests, and a tmp subdirectory is created in it.
		echo ': Starting container'
		# This is assumed to have abundant free space for project
		# clones and environment
		CONT_TMP=$(mktemp -d)
		echo "
		# Commands to run inside singularity
		# /work is assumed to contain a copy of the project (which is *not* the one
		# being tested, but usually a bootstrap copy containing this very script)
		set -eu
		echo ':: Recursing in container:/work -> host:$(realpath $(pwd)))'
		cd /work
		tests/test.sh tmp-clone
		" | singularity exec -C \
			-B "$(realpath $(pwd)):/work" \
			-B "${GH_PAT_FILE}:/pat_file" \
			-B "${CONT_TMP}:/tmp" \
			'docker://python:3.10.10' bash
		;;
	tmp-clone)
		DIR=$(mktemp -d)
		echo "::: Cloning repository"
		git config --global url."https://ssh:$(</pat_file)@github.com/".insteadOf 'ssh://git@github.com/'
		git config --global url."https://git:$(</pat_file)@github.com/".insteadOf 'git@github.com:'
		git clone --recurse-submodules 'git@github.com:emorice/gemz' "$DIR/gemz"
		echo "::: Creating environment"
		python -m venv "$DIR/venv"
		. "$DIR/venv/bin/activate"
		pip --no-cache-dir install wheel
		echo "::: Installing dependencies"
		pip --no-cache-dir install "$DIR/gemz[test]" "$DIR/gemz/gemz-galp[test]"
		echo "::: Running tests"
		pytest "--ignore=$DIR" -k 'not peer and not nonlinear_shrinkage' -v
		;;
	*)
		echo "Unknown, or no subcommand given"
esac

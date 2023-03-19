#!/bin/bash
#SBATCH -p hh
#SBATCH --mail-type=END,FAIL

set -eu
module load singularity
tests/test.sh singularity


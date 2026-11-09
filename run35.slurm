#!/usr/bin/env bash
#SBATCH --job-name=psor35
#SBATCH --output=psor35_%j.out
#SBATCH --partition=short-28core
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=28
#SBATCH --exclusive
#SBATCH --time=03:30:00

# AMS 530 Project 3, Problem 3.5
# 2-asset American option pricing by projected SOR.
#
# Three nodes x 28 cores = 84 slots, which is what P = 64 needs.
# --exclusive matters: on Project 2 a shared node produced a spurious
# dip in the speedup curve that took a replication run to rule out.
#
# NOTE: do NOT add "module purge" here. On the legacy login nodes it
# strips slurm from the environment and the job dies with
# "sbatch: command not found".

set -u
cd "$SLURM_SUBMIT_DIR"

echo "=============================================================="
echo "job      : $SLURM_JOB_ID"
echo "nodes    : $SLURM_JOB_NODELIST"
echo "started  : $(date)"
echo "=============================================================="

# ---- MPI module: take the first one that actually provides mpicc ----
for m in mvapich2/gcc12.1/2.3.7 mvapich2/gcc/2.3.7 mvapich2 \
         openmpi/gcc/4.1.5 openmpi/gcc openmpi; do
    module load "$m" 2>/dev/null && command -v mpicc >/dev/null 2>&1 && {
        echo "loaded MPI module: $m"; break; }
done

if ! command -v mpicc >/dev/null 2>&1; then
    echo "ERROR: no mpicc on PATH. Run 'module avail' and edit the list above."
    exit 1
fi

echo "mpicc    : $(command -v mpicc)"
mpicc -show 2>/dev/null | head -1
echo

# ---- build ----
mpicc -O3 -std=c99 -Wall -o psor2d psor2d.c -lm || { echo "BUILD FAILED"; exit 1; }
echo "build OK"
echo

NT=50                       # time steps (backward Euler)
MESHES="128 256 512 1024"   # interior points per dimension
PROCS="1 2 4 16 64"

# ---- validation: European runs have a closed-form answer ----
echo "######## VALIDATION (European, projection disabled) ########"
for M in $MESHES; do
    mpirun -np 1 ./psor2d -M "$M" -n "$NT" -e
done
echo

# ---- validation: the American answer must not depend on P ----
echo "######## CONSISTENCY (American, M=256, answer must match) ########"
for P in $PROCS; do
    mpirun -np "$P" ./psor2d -M 256 -n "$NT"
done
echo

# ---- performance sweep ----
echo "######## PERFORMANCE SWEEP (American) ########"
for M in $MESHES; do
    for P in $PROCS; do
        echo "---- M=$M P=$P ----"
        mpirun -np "$P" ./psor2d -M "$M" -n "$NT"
    done
done

echo
echo "=============================================================="
echo "finished : $(date)"
echo "=============================================================="

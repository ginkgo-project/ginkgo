#!/usr/bin/env bash

# Submits a Slurm batch job, streams its output and waits until it finished.
# Usage: sbatch-wait.sh [sbatch options] script [script arguments]
#
# Exits with the exit code of the batch job. If this script is terminated,
# e.g. because the CI job was cancelled, the Slurm job is cancelled as well,
# so it does not keep running (and consuming compute time) in the background.

name="slurm-${CI_JOB_ID:-$$}"
output="${name}.out"
job_id_file="${name}.id"
rm -f "${output}" "${job_id_file}"
touch "${output}"

# with --wait, sbatch prints the job id right after the submission and then
# only returns when the job has finished, with the exit code of the job
sbatch --wait --parsable --output="${output}" "$@" > "${job_id_file}" &
sbatch_pid=$!

while [[ ! -s "${job_id_file}" ]] && kill -0 "${sbatch_pid}" 2> /dev/null; do
  sleep 1
done
job_id=$(cut -d ';' -f 1 "${job_id_file}")
if [[ -z "${job_id}" ]]; then
  wait "${sbatch_pid}"
  echo "Job submission failed"
  exit 1
fi
echo "Submitted Slurm job ${job_id}, waiting for it to start and finish"

tail -n +1 -F "${output}" 2> /dev/null &
tail_pid=$!

# called through the trap below
# shellcheck disable=SC2317
cancel() {
  echo "Cancelling Slurm job ${job_id}"
  scancel "${job_id}"
  kill "${tail_pid}" 2> /dev/null
  exit 1
}
trap cancel INT TERM

wait "${sbatch_pid}"
status=$?

# give tail a moment to print the last lines of the output file
sleep 5
kill "${tail_pid}" 2> /dev/null
echo "Slurm job ${job_id} finished with exit code ${status}"
exit "${status}"

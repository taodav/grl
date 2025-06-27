import subprocess
import multiprocessing
import argparse
import os
import sys

def worker(worker_id: int, sweep_id: str, project_root: str, count: int, job_id: str):
    """
    This function is executed by each worker process.
    It runs a wandb agent and redirects its output to a unique log file.
    """
    pid = os.getpid()
    print(f"Worker {worker_id} (PID: {pid}) starting agent for sweep: {sweep_id}")

    # --- Command setup ---
    command = [
        "python",
        "-m",
        "wandb",
        "agent",
        sweep_id
    ]
    # If count is positive, add it to the command to limit the number of runs.
    if count > 0:
        command.extend(["--count", str(count)])
        print(f"Worker {worker_id} will run for {count} experiment(s).")
    else:
        print(f"Worker {worker_id} will run continuously.")


    # --- Logging setup ---
    # Create a dedicated directory for worker logs to keep things tidy.
    worker_log_dir = os.path.join(project_root, "logs", f"workers_{job_id}")
    os.makedirs(worker_log_dir, exist_ok=True)

    # Define unique output and error log files for this worker.
    stdout_log_path = os.path.join(worker_log_dir, f"worker_{worker_id}_pid_{pid}.out")
    stderr_log_path = os.path.join(worker_log_dir, f"worker_{worker_id}_pid_{pid}.err")

    try:
        # Open the log files
        with open(stdout_log_path, 'w') as stdout_log, open(stderr_log_path, 'w') as stderr_log:
            # Run the agent command, redirecting stdout and stderr to our log files.
            subprocess.run(
                command,
                cwd=project_root,
                check=True,
                stdout=stdout_log,
                stderr=stderr_log
            )
    except subprocess.CalledProcessError as e:
        print(f"Worker {worker_id} (PID: {pid}) failed with error: {e}")
    except KeyboardInterrupt:
        print(f"Worker {worker_id} (PID: {pid}) received interrupt. Exiting.")
    finally:
        print(f"Worker {worker_id} (PID: {pid}) finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple W&B agents in parallel on a single machine.")
    parser.add_argument("--sweep_id", type=str, required=True, help="The W&B sweep ID to connect to.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel agents to launch.")
    parser.add_argument("--project_root", type=str, default=".", help="The root directory of the project.")
    parser.add_argument("--count", type=int, default=0, help="Number of runs per agent. Use 0 to run continuously until the sweep is complete.")
    parser.add_argument("--job_id", type=str, required=True, help="The Slurm Job ID for creating unique log directories.")
    args = parser.parse_args()

    processes = []

    print(f"Starting manager for sweep {args.sweep_id} with {args.num_workers} workers.")

    try:
        # Launch each worker in its own process, passing the required arguments.
        for i in range(args.num_workers):
            p = multiprocessing.Process(target=worker, args=(i, args.sweep_id, args.project_root, args.count, args.job_id))
            processes.append(p)
            p.start()

        # Wait for all worker processes to complete
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\nManager received interrupt. Terminating all workers.")
        for p in processes:
            p.terminate()
            p.join()

    print("All workers have finished. Manager exiting.")

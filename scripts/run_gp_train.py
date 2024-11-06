import multiprocessing
import subprocess

def run_task(command):
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    commands = [
        "python launch.py gp_train.py gp_uq_normal_autoreg.yaml --port 29501",
        "python launch.py gp_train.py gp_uq_normal_excg.yaml --port 29502",
        ]

    processes = []
    for command in commands:
        p = multiprocessing.Process(target=run_task, args=(command,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All tasks completed.")
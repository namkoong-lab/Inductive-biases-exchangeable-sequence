import multiprocessing
import subprocess

def run_task(command):
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    commands = [
        "python launch.py train.py uq_normal_autoreg_mix.yaml --port 29503",
        "python launch.py train.py uq_normal_excg_mix.yaml --port 29504",
        ]

    processes = []
    for command in commands:
        p = multiprocessing.Process(target=run_task, args=(command,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All tasks completed.")
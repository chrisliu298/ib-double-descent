import os
import subprocess
import sys
from multiprocessing import Pool
from typing import List

from data import MNISTDataModule, FashionMNISTDataModule

"""
A utility for running multiple experiments on multiple GPUs
"""

if __name__ == '__main__':
    PER_GPU_COUNT: int = 32
    print(os.getcwd())
    EXPMT_DIR: str = sys.argv[1]
    assert EXPMT_DIR is not None, "un-specified experiment directory"
    matches: List[str] = [f for f in os.listdir(EXPMT_DIR)]
    matches = [m for m in matches if m not in ('',)]  # leaving empty
    SCRIPT: str = "train.py"

    def worker(config_file: str):
        # working with GPU 0 modify as needed
        process = subprocess.Popen(['../../venv/bin/python',
                                    SCRIPT,
                                    f"{EXPMT_DIR}/{config_file}"],
                                   )
        process.wait()


    # pre-download so that when we load in each new process, we don't re-download
    MNISTDataModule({}).prepare_data()
    FashionMNISTDataModule({}).prepare_data()
    with Pool(PER_GPU_COUNT) as p:
        p.map(worker, matches)



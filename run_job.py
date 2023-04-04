#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import sys, os
from utils.runner import start_run


if __name__ == "__main__":
    run_id = sys.argv[1]
    run_type = sys.argv[2]
    start_run(run_type, run_id)

#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import sys, os
import run


if __name__ == "__main__":
    run_id = sys.argv[1]
    run_type = sys.argv[2]
    run.start_run(run_type, run_id)

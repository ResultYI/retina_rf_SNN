from pathlib import Path
import subprocess
import sys
import time

import psutil

from common import OUT, ROOT


if __name__ == '__main__':
    training_pid = int(sys.argv[1])
    while not (OUT / 'checkpoints/training_complete.json').exists():
        if not psutil.pid_exists(training_pid):
            raise RuntimeError('Training process ended before its completion lock; inspect the training log')
        time.sleep(10)
    for script in ('analyze.py', 'report.py'):
        print('START', script, flush=True)
        arguments = ['--workers', '2'] if script == 'analyze.py' else []
        subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).with_name(script)), *arguments],
                       cwd=ROOT, check=True)
    print('PIPELINE COMPLETE', flush=True)

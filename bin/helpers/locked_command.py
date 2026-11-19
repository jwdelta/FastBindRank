#!/usr/bin/env python3

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys


def check_lock(fd, path):
    """Verify an inherited descriptor belongs to the expected lock and holds it."""
    actual, expected = os.fstat(fd), Path(path).stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Inherited lock does not match the requested directory')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock-file', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    path = Path(a.lock_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f'Another preparation is running: {path.parent}')
        # The child inherits the lock, so it stays held even if this supervisor exits.
        result = subprocess.run(command + ['--lock-fd', str(lock.fileno())],
                                pass_fds=(lock.fileno(),))
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

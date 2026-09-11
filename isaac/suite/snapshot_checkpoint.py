"""Copy a completed Torch ZIP checkpoint without racing the training writer."""

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


def snapshot(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    fd, path = tempfile.mkstemp(dir=destination.parent, suffix=".snapshot")
    os.close(fd)
    try:
        shutil.copyfile(source, path)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(
                "Checkpoint changed during copying; retry after the write finishes"
            )
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Checkpoint ZIP integrity check failed")
        os.replace(path, destination)
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    snapshot(a.source, a.destination)

import glob
import os

import numpy as np

class TrajCollector:
    def __init__(
            self,
            trj_dir: str,
            pattern: str = "*.nc",
            exclude_last: bool = False,
    ):
        self.trj_dir = trj_dir
        self.pattern = pattern
        self.exclude_last = exclude_last

    def collect(self) -> list[str]:
        files = sorted(
            glob.glob(os.path.join(self.trj_dir, self.pattern)),
        )

        if not files:
            raise FileNotFoundError(
                f'no files matching {self.pattern} '
                f'found in {self.trj_dir}'
            )

        if self.exclude_last and len(files) > 1:
            excluded_files = files[-1]
            files = files[:-1]
            print('Excluded file: ', excluded_files)

        return files

class DistanceWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def write(
            self,
            frames: np.ndarray,
            distances: np.ndarray,
    ) -> None:
        np.savetxt(
            self.output_path,
            np.column_stack([frames, distances]),
            header='Frame\tDistance',
        )
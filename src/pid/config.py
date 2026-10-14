import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    trj_dir: str
    ref_path: str
    selection: str
    start_frame: int
    end_frame: int
    workers: int
    output_dir: str
    prefix: str
    threshold: float
    exclude_last_trajectory: bool = False

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def dat_path(self) -> str:
        return os.path.join(self.output_dir, f'{self.prefix}_distance.dat')

    @property
    def png_path(self) -> str:
        return os.path.join(self.output_dir, f'{self.prefix}_histogram.png')
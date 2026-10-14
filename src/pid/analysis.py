import multiprocessing
import itertools
import os
from multiprocessing import Pool


import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

from pid.blocks import FrameBlock, FrameBlockAllocator
from pid.config import Config
from pid.plotting import HistPlotter
from pid.utils import TrajCollector, DistanceWriter


class ShiftsPeriodicImages:
    def __init__(self):
        Shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=float)

        self.shifts = Shifts[np.any(Shifts != 0, axis=1)]

    def Cartesian_offsets(self, box_vectors: np.ndarray) -> np.ndarray:
        return self.shifts @ box_vectors

class CountPeriodicImagesDistance:
    def __init__(
            self,
            ref_path: str,
            trj_files: list[str],
            selection: str = 'protein',
    ):
        self.ref_path = ref_path
        self.trj_files = trj_files
        self.selection = selection

        self.shifts = ShiftsPeriodicImages()

    def _calculate_frame(self, protein, timestep) -> float:
        box_vectors = timestep.triclinic_dimensions
        image_offsets = self.shifts.Cartesian_offsets(box_vectors)

        min_dist = np.inf

        for offset in image_offsets:
            image_positions = protein.positions + offset

            distance_matrix = distance_array(
                protein.positions,
                image_positions,
            )

            min_dist = min(min_dist, distance_matrix.min())

        return min_dist

    def calculate_block(self, block: FrameBlock) -> tuple[np.ndarray, np.ndarray]:
        universe = mda.Universe(
            self.ref_path,
            self.trj_files,
        )

        protein = universe.select_atoms(self.selection)

        frames = np.arange(block.start_frame, block.end_frame)
        distances = np.empty(block.size)

        trj_slice = universe.trajectory[block.start_frame:block.end_frame]

        for i, timestep in enumerate(
            tqdm(
                trj_slice,
                total=block.size,
                desc=f'Frames {block.start_frame}-{block.end_frame}',
                leave=False,
            )
        ):
            distances[i] = self._calculate_frame(
                protein=protein,
                timestep=timestep,
            )
        return frames, distances

class DistanceResults:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.distances: list[np.ndarray] = []

    def add_block(
            self,
            frames: np.ndarray,
            distances: np.ndarray,
    ) -> None:

        self.frames.append(frames)
        self.distances.append(distances)

    def finale(self) -> tuple[np.ndarray, np.ndarray]:
        frames = np.concatenate(self.frames)
        distances = np.concatenate(self.distances)

        order = np.argsort(frames)

        return frames[order], distances[order]


class PeriodicImageAnalysis:
    def __init__(
            self,
            config: Config
    ):
        self.config = config

        self.trj_files = TrajCollector(
            trj_dir=self.config.trj_dir,
            exclude_last=self.config.exclude_last_trajectory,
        ).collect()

        self._validate_frame_range()

        self.calculator = CountPeriodicImagesDistance(
            ref_path=self.config.ref_path,
            trj_files=self.trj_files,
            selection=self.config.selection,

        )

        self.results = DistanceResults()

    def _validate_frame_range(self) -> None:
        universe = mda.Universe(
            self.config.ref_path,
            self.trj_files,
        )

        n_validate = universe.trajectory.n_frames

        if self.config.end_frame > n_validate:
            raise ValueError(
                f'--stop={self.config.end_frame} exceeds available '
                f'number of frames {n_validate}'
            )

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        allocator = FrameBlockAllocator(
            start=self.config.start_frame,
            end=self.config.end_frame,
            n_workers=self.config.workers
        )

        blocks = allocator.allocate()

        tasks = [
            (self.config, block, self.trj_files)
            for block in blocks
        ]

        if len(tasks) == 1:
            results = [_worker_calculate_block(tasks[0])]
        else:
            with Pool(processes=len(tasks)) as pool:
                results = pool.map(_worker_calculate_block, tasks)

        for frame, distances in results:
            self.results.add_block(frame, distances)

        return self.results.finale()

    def save(self, frames, distances) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)

        DistanceWriter(self.config.dat_path).write(frames=frames, distances=distances)

        print(f"Saved distances: {self.config.dat_path}")

    def plot(self, distances) -> None:
        plotter = HistPlotter(
            output_path=self.config.png_path,
            threshold=self.config.threshold,
        )

        plotter.plot(distances)


def _worker_calculate_block(task):
    config, block, trj_files = task

    calculator = CountPeriodicImagesDistance(
        ref_path=config.ref_path,
        trj_files=trj_files,
        selection=config.selection,
    )

    return calculator.calculate_block(block)
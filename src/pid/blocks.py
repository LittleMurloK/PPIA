from dataclasses import dataclass

@dataclass(frozen=True)
class FrameBlock:
    start_frame: int
    end_frame: int

    @property
    def size(self) -> int:
        return self.end_frame - self.start_frame


class FrameBlockAllocator:
    def __init__(self, start: int, end: int, n_workers: int):
        if end <= start:
            raise ValueError('end must be larger than start')

        if n_workers < 1:
            raise ValueError('n_workers must be at least 1')

        self.start = start
        self.end = end
        self.n_frames = end - start
        self.n_workers = min(n_workers, self.n_frames)

    def allocate(self) -> list[FrameBlock]:
        base_size = self.n_frames // self.n_workers
        remainder = self.n_frames % self.n_workers

        blocks = []
        current_frame = self.start

        for worker_id in range(self.n_workers):
            block_size = base_size + (1 if worker_id < remainder else 0)

            if block_size == 0:
                continue

            block = FrameBlock(
                start_frame=current_frame,
                end_frame=current_frame + block_size,
            )

            blocks.append(block)

            current_frame = block.end_frame

        return blocks
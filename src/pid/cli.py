import argparse

from pid.analysis import PeriodicImageAnalysis
from pid.config import Config

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the minimum distance between a protein and "
            "its periodic images for each trajectory frame."
        )
    )
    parser.add_argument(
        '-t',
        "--trj_dir",
        required=True,
        help="Path to the directory with trajectories",
    )

    parser.add_argument(
        '-r',
        "--reference",
        required=True,
        help="Path to the reference structure",

    )

    parser.add_argument(
        '-c',
        "--selection",
        default="protein",
        help="MDAnalysis atom selection. Default: protein",
    )

    parser.add_argument(
        '-s',
        "--start",
        type=int,
        default=0,
        help="Start frame index. Default: 0",
    )

    parser.add_argument(
        '-e',
        "--end",
        type=int,
        required=True,
        help="End frame index.",
    )

    parser.add_argument(
        '-w',
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Default: 1",
    )

    parser.add_argument(
        '-o',
        "--output_dir",
        default="./out",
        help="Path to the output directory. Default: ./out",
    )

    parser.add_argument(
        '-p',
        "--prefix",
        default="PPI",
        help="Prefix for output files. Default: PPI",
    )

    parser.add_argument(
        '-b',
        "--threshold",
        type=float,
        default=5.0,
        help="Minimum distance threshold. Default: 5.0",
    )

    parser.add_argument(
        '-x',
        "--exclude_last",
        action="store_true",
        help="Exclude last trajectories. Default: False",
    )

    args = parser.parse_args()

    return Config(
        ref_path=args.reference,
        trj_dir=args.trj_dir,
        selection=args.selection,
        start_frame=args.start,
        end_frame=args.end,
        workers=args.workers,
        output_dir=args.output_dir,
        prefix=args.prefix,
        threshold=args.threshold,
        exclude_last_trajectory=args.exclude_last,
    )

def main():
    config = parse_args()

    analysis = PeriodicImageAnalysis(config)

    frames, distances = analysis.run()

    analysis.save(frames, distances)
    analysis.plot(distances)

if __name__ == "__main__":
    main()
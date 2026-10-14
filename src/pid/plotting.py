import matplotlib.pyplot as plt
import numpy as np

class HistPlotter:

    def __init__(
            self,
            output_path: str,
            threshold: float = 5.0,
            n_bins: int = 50,
    ):
        self.output_path = output_path
        self.threshold = threshold
        self.n_bins = n_bins

    def plot(
            self,
            distances: np.ndarray,
    ) -> None:

        histogram, bin_edges = np.histogram(distances, bins=self.n_bins, density=True)

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]

        fraction_below = (
            np.sum(distances < self.threshold)
            / distances.size
            * 100.0
        )

        fig, ax = plt.subplots(figsize = (10,5))

        ax.bar(
            bin_centers,
            histogram,
            width = bin_width,
            edgecolor = "black",
            color = "crimson",
            alpha = 0.8,
        )

        ax.axvline(
            self.threshold,
            lw=3,
            color='royalblue',
        )

        ax.text(0.98, 0.95,
                fr"< {self.threshold} $\AA$ {fraction_below:.3f}%",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=22,
                color="royalblue",
                )

        ax.set_xlim(bin_edges[0] - 1, bin_edges[-1] + 1)
        ax.set_xlabel(r"Distance, $\AA$")
        ax.set_ylabel(r"$P(\mathrm{distance})$", fontsize=18)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.grid(True, color="gray", linewidth=0.7, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=1000)
        plt.close(fig)
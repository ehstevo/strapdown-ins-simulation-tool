from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_dir, filename, dpi=300, save_pdf=True):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if save_pdf:
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    else:
        png_path = output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")


def configure_axes(ax, xlabel=None, ylabel=None, title=None, equal=False, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if equal:
        ax.axis("equal")
    if grid:
        ax.grid(True, alpha=0.3)

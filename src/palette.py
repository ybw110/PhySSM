"""Unified low-saturation Teal-Cyan-Blue & Muted Terracotta-Amber palette for Paper 1 Experiments (Nature/NC style)."""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Unified publication palette across all benchmark and mechanism visualizations
PAL = {
    # Primary Teal / Cyan-Blue family (cool, calm, dominant)
    "deep_teal": "#26708B",      # Primary dark cyan-blue
    "steel_blue": "#528DAD",     # Muted Morandi steel blue
    "sky_blue": "#81AFD4",       # Soft sky blue
    "cyan_teal": "#6FB1B9",      # Muted cyan-teal
    "sage_teal": "#6C9D8E",      # Muted sage-cyan
    "mist_blue": "#9EBDCF",      # Soft mist blue
    "pale_cyan": "#C4E0E8",      # Pale cyan bar/band fill
    "ice_blue": "#D5E8F1",       # Whisper ice-blue band fill

    # Accent Muted Red-Orange-Amber family (warm highlights, Ours, tails)
    "terracotta": "#C16852",     # Muted terracotta red-orange (Ours / primary warm accent)
    "muted_coral": "#DE7C7D",    # Low-saturation rose-coral
    "dusty_apricot": "#DAA799",  # Dusty rose-apricot
    "soft_peach": "#FBCFBB",     # Pale peach-coral fill
    "soft_amber": "#E2B87E",     # Low-saturation warm amber-yellow
    "pale_amber": "#F3E1C0",     # Pale warm sand fill

    # Neutral ink, grid, and subtle backgrounds
    "ink": "#2B3A42",
    "slate_gray": "#7B8B95",
    "light_slate": "#B5C2CA",
    "grid": "#CFD8DF",
    "train_bg": "#EFF6F9",
    "val_bg": "#FCF6EC",
    "test_bg": "#FBF0EE",
}

MODEL_STYLE = {
    "Composite-GPR": {
        "color": "#8B9AA4",
        "fill": "#DDE4E8",
        "ls": ":",
        "lw": 1.45,
        "marker": "v",
        "family": "Traditional SHM",
    },
    "N4SID": {
        "color": "#6C9D8E",
        "fill": "#CFE3DC",
        "ls": "-.",
        "lw": 1.50,
        "marker": "d",
        "family": "Control SysID",
    },
    "VMD-BiLSTM": {
        "color": "#81AFD4",
        "fill": "#D5E8F1",
        "ls": "--",
        "lw": 1.50,
        "marker": "^",
        "family": "Radio Telescope / Ind.",
    },
    "Causal-Dilated TCN": {
        "color": "#6FB1B9",
        "fill": "#C4E0E8",
        "ls": "--",
        "lw": 1.55,
        "marker": "s",
        "family": "SHM / MSSP",
    },
    "ST-GAT": {
        "color": "#528DAD",
        "fill": "#B9D3E3",
        "ls": "-.",
        "lw": 1.55,
        "marker": "P",
        "family": "IEEE TIM",
    },
    "TimeXer": {
        "color": "#E2B87E",
        "fill": "#F3E1C0",
        "ls": "--",
        "lw": 1.65,
        "marker": "D",
        "family": "NeurIPS 2024",
    },
    "TimePro": {
        "color": "#26708B",
        "fill": "#9EBDCF",
        "ls": "-",
        "lw": 1.75,
        "marker": "o",
        "family": "ICML 2025 (SSM)",
    },
    "PhySSM": {
        "color": "#C16852",
        "fill": "#FBCFBB",
        "ls": "-",
        "lw": 2.15,
        "marker": "*",
        "family": "Ours (Physics SSM)",
    },
}

NATURE_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nature_teal_terracotta",
    [
        "#26708B",
        "#528DAD",
        "#88A9C9",
        "#C4E0E8",
        "#F7F9FA",
        "#F6DCD7",
        "#E9A9AC",
        "#DE7C7D",
        "#C16852",
    ],
    N=256,
)


def apply_nature_rcparams() -> None:
    """Apply unified publication typography and font sizes across all figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11.2,
        "axes.titlesize": 12.8,
        "axes.titleweight": "bold",
        "axes.titlepad": 7.5,
        "axes.labelsize": 12.0,
        "axes.labelweight": "bold",
        "legend.fontsize": 10.0,
        "xtick.labelsize": 10.6,
        "ytick.labelsize": 10.6,
        "axes.linewidth": 1.0,
        "axes.edgecolor": PAL["ink"],
        "text.color": PAL["ink"],
        "axes.labelcolor": PAL["ink"],
        "xtick.color": PAL["ink"],
        "ytick.color": PAL["ink"],
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

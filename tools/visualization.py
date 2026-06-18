import matplotlib.pyplot as plt

TOL_PALETTE = [
    "#332288", "#117733", "#44AA99", "#88CCEE",
    "#DDCC77", "#CC6677", "#AA4499", "#888888"
]

VARIANT_COLORS = {
    "nano": "#2ecc71",
    "base": "#3498db",
    "pro": "#e67e22",
    "promax": "#e74c3c",
    "unet_lite_no_attn": "#3498db",
    "unet_lite_with_attn": "#e67e22",
    "unet_mobilevit": "#e74c3c",
}

def configure_paper_style():
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'grid.alpha': 0.3,
        'axes.grid': True,
    })

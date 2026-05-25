from matplotlib import pyplot as plt

LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14

save_default = "../../assets"

plt.rcParams.update({
    "font.size": LABEL_FONTSIZE,                      
    "axes.titlesize": 14,                 
    "axes.labelsize": LABEL_FONTSIZE,                 
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": LEGEND_FONTSIZE,
    "axes.linewidth": 1.2,                
    "grid.alpha": 0.5                   
})
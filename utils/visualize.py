import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_heatmap(
    data, 
    label_x, label_y, 
    label_x_name, label_y_name,
    title):        
    
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(label_x)))
    ax.set_yticks(np.arange(len(label_y)))
    ax.set_xticklabels(label_x)
    ax.set_yticklabels(label_y)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
            rotation_mode="anchor")
    ax.set_title(title)
    fig.tight_layout()

    return fig
    
    
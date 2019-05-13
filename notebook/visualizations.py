import matplotlib.pyplot as plt
import pandas as pd

def visualize_target_by_identity(data, columns):
    data.groupby(data["target"] >= .5).mean()[columns].T \
        .plot.barh()

    plt.legend(title="Is offensive")

    plt.tight_layout()
    plt.title("Comment summary – targeted identity")
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def visualize_toxicity_by_identity(data, columns):
    """Visualizes the toxicity for each identity type"""
    data.groupby(data["target"] >= .5).mean()[columns].T \
        .plot.barh()

    plt.legend(title="Is toxic")

    plt.tight_layout()
    plt.title("Comment summary – targeted identity")
    plt.show()


def visualize_target_distribution(data, bins = 10):
    """
    Visualizes the target distribution
    - A comment is considered non-toxic if target is less than .1
    - Comments between .1 to .5 are considered toxic to some annotators
    - Comments above .5 are toxic.
    """
    plot = data["target"].plot(kind="hist", bins=bins)

    ax = plot.axes

    for p in ax.patches:
        ax.annotate(
            f'{p.get_height() * 100 / data.shape[0]:.2f}%',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="center", 
            fontsize=8, 
            color="black",
            xytext=(0,7), 
            textcoords="offset points"
        )
    
    plt.title("Target distribution")
    plt.show()


def visualize_comment_length(data, title):
    """Visualizes the comment length distribution"""
    data["comment_text_len"] = data["comment_text"].apply(len)
    ax = sns.distplot(
        data["comment_text_len"],
        bins=150
    )
    ax.set(xlabel="length", ylabel="count")

    plt.title(title)
    plt.show()

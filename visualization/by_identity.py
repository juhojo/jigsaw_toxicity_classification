import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./notebook/train_cleaned.csv")

identity_columns = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness"
]

data.groupby(data["target"] >= .5).mean()[identity_columns].T \
    .plot.barh()

plt.legend(title="Is offensive")

plt.tight_layout()
plt.title("Comment summary – targeted identity")
plt.show()

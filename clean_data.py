import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import seaborn as sns
import umap


def get_us_rank(s):
    """Get US ranking from long string"""
    m = re.search(r"#\d+", s)
    if m:
        return int(m.group(0)[1:])
    else:
        return np.nan


shoe_data_raw = pd.read_csv(
    os.path.join("data", "RRRShoes.csv"),
    encoding="cp1252"
    )
shoe_data = shoe_data_raw.dropna(how="all")
shoe_data["USRankNum"] = shoe_data["USARanking"].apply(get_us_rank)
feat_names = [
    "TotalScore",
    "TotalReviews",
    "Price",
    "ShoeWeight",
    "ToeDrop",
    "FootHeight",
    "USRankNum",
    "FiveStars",
    "FourStars",
    "ThreeStars",
    "TwoStars",
    "OneStars"
]

cont_raw = shoe_data[feat_names].dropna()
cont_data = cont_raw.values
X_scaled = pre.StandardScaler().fit_transform(cont_data)
red = umap.UMAP()
emb = red.fit_transform(X_scaled)

sns.set(style="white", rc={"figure.figsize": (8, 8)})
plt.scatter(
    emb[:, 0],
    emb[:, 1],
    s=(max(cont_raw["USRankNum"]) - cont_raw["USRankNum"]) / 10,
    c=cont_raw["TotalScore"],
    cmap="RdYlGn"
    )
plt.show()

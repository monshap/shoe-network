import os
import re
import itertools as it

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.preprocessing as pre
import umap
from sklearn.impute import KNNImputer
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.metrics import pairwise_distances


def purp_to_num(se):
    """Convert purpose string to numeric code"""
    se_list = se.apply(lambda x: x.split("; "))
    tempo = se_list.apply(lambda x: "Tempo" in x).astype(int)
    competition = se_list.apply(lambda x: "Competition" in x).astype(int)
    daily = se_list.apply(lambda x: "Daily" in x).astype(int)
    df = pd.DataFrame({
        "Tempo": tempo,
        "Competition": competition,
        "Daily": daily
    })
    return df


def add_rater_edges(G, df, col, thalf=2, year=2025):
    """Add edges between rater and shoes they rated"""
    G.add_node(col)
    se = df[col]
    for idx, val in se.dropna().items():
        decay = np.exp(-np.log(2)/thalf * (year - df.loc[idx, "Year"]))
        G.add_edge(col, idx, weight=decay*val)
    return G


def add_rater_sim_edges(G, df, pair):
    """Add edges between similar raters"""
    r1 = pair[0]
    r2 = pair[1]
    sub_df = df[[r1, r2]].dropna()
    dist = np.sqrt(np.sum((sub_df[r1] - sub_df[r2])**2))/len(sub_df)
    sim = (1 - dist)  # normalize to [0, 1]
    G.add_edge(r1, r2, weight=sim)
    return G

grade_map = {
    "A+": 2.5,
    "A": 2.0,
    "A-": 1.5,
    "B+": 1.0,
    "B": 0.5,
    "B-": 0.0,
    "C+": -0.5,
    "C": -1.0,
    "C-": -1.5,
    "F": -2
}

shoe_data_raw = pd.read_csv(
    os.path.join("data", "ManualRRData.csv"),
    encoding="utf-8"
)
shoe_data = shoe_data_raw.dropna(subset="TotalScore")
purp_cols = purp_to_num(shoe_data["Purpose"].dropna())
shoe_data = pd.merge(shoe_data, purp_cols, left_index=True, right_index=True,
                     how="left")
cont_var_names = [
    "ShockAbs",
    "EnergyReturn",
    "HeelStack",
    "ForefootStack",
    "Drop",
    "MidsoleSoftness",
    "Width",
    "ToeboxWidth",
    "Flexibility",
    "ColdStiffChange",
    "Weight",
    "HeelCounterStiff",
    "ForeWidth",
    "HeelWidth",
    "ColdSoftChange",
    "Year",
    "Tempo",
    "Competition",
    "Daily"
]
cont_vars_wna = shoe_data[cont_var_names]
imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(cont_vars_wna.values)
cont_vars_imputed = cont_vars_wna.copy()
cont_vars_imputed.loc[:,:] = X_imputed
X_scaled = pre.StandardScaler().fit_transform(X_imputed)

reducer = umap.UMAP(n_neighbors=4, min_dist=0.2)
emb = reducer.fit_transform(X_scaled)
cont_vars_imputed.loc[:, "emb0"] = emb[:, 0]
cont_vars_imputed.loc[:, "emb1"] = emb[:, 1]
px_data = pd.merge(
    shoe_data,
    cont_vars_imputed[["emb0", "emb1"]],
    left_index=True,
    right_index=True,
    how="inner"
)

knn_graph = kneighbors_graph(emb, 1, mode="distance", include_self=False)
radius_graph = radius_neighbors_graph(emb, 1.5, mode="distance",
                                      include_self=False)
G0 = nx.from_scipy_sparse_array(radius_graph)
# G = nx.Graph()
# G.add_nodes_from(range(len(shoe_data)))
raters = ["Monica", "Andrea", "Matt", "David", "Nathan"]
for rater in raters:
    px_data.loc[:, rater] = shoe_data.loc[:, rater].map(grade_map)
    G = add_rater_edges(G0, px_data, rater)
for rpair in it.combinations(raters, 2):
    G = add_rater_sim_edges(G, px_data, rpair)

nxmap = {i: shoe_data.loc[i, "ShoeName"] for i in range(len(shoe_data))}
G_labeled = nx.relabel_nodes(G, nxmap)
init_pos = {
    **{shoe: (emb[i,0], emb[i,1]) for i, shoe in nxmap.items()},
    **{rater: np.mean(emb, axis=0) for rater in raters}
}

pos = nx.spring_layout(G_labeled, pos=init_pos, fixed=list(nxmap.values()))
# pos = nx.spring_layout(G_labeled, pos=pos)
for i, shoe in nxmap.items():
    px_data.loc[i, "pos0"] = pos[shoe][0]
    px_data.loc[i, "pos1"] = pos[shoe][1]
rater_pos = [(rater, pos[rater][0], pos[rater][1]) for rater in raters]
r_data = pd.DataFrame(rater_pos, columns=["Name", "x", "y"])

fig = px.scatter(
    px_data,
    x="pos0",
    y="pos1",
    color="TotalScore",
    color_continuous_scale="YlGn",
    hover_name="ShoeName",
    hover_data={
        "pos0": False,
        "pos1": False,
        "Brand": True,
        "TotalScore": True,
        "Price": ":.2f",
        "Year": True
        }
    )
fig.update_traces(marker=dict(size=20))
fig.add_trace(
    go.Scatter(
        x=r_data["x"],
        y=r_data["y"],
        text=r_data["Name"],
        mode="markers+text",
        hoverinfo="skip",
        showlegend=False,
        marker=dict(size=50, symbol="square", color="LightBlue")
    )
)
fig.show()

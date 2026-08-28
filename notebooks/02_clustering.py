# %% [markdown]
# # Clustering: Does a Renewable Site Typology Emerge?
#
# The EDA notebook hand-labeled 13 sites into 6 climate zones (Tell,
# High Plateau, Saharan Atlas, ...). That labeling was based on
# geography, not on the actual renewable-relevant behavior of each site.
#
# This notebook asks the real question: **if we cluster purely on
# resource strength, predictability, solar/wind complementarity, and heat
# exposure, with no location information at all, do we recover the
# same zones, or does a different, more decision-relevant grouping show
# up?**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_theme(style="whitegrid")
import os
FIG_DIR = "../reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("../data/processed/site_summary_features.csv", index_col=0)
df

# %% [markdown]
# ## 1. Feature selection
#
# Deliberately **excluding lat/lon and zone** from the clustering
# features, since those are the labels we're testing against, not inputs.
# Also dropping features that are near-duplicates of another one already
# in the set (WS10M vs. WS50M; DNI vs. DWN). Kept the hub-height wind
# speed and the flat-panel-relevant GHI since flat-panel PV is the more
# likely near-term Algerian deployment vs. CSP.

# %%
corr = df[
    ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "WS10M", "WS50M", "T2M",
     "mean_kt", "std_kt", "interannual_cv_pct", "solar_wind_monthly_corr",
     "hot_days_per_year"]
].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature correlation (raw, pre-selection)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/clustering_feature_correlation.png", dpi=150)
plt.show()

# %%
FEATURES = [
    "ALLSKY_SFC_SW_DWN",       # resource strength
    "WS50M",                  # wind resource strength
    "mean_kt",                # predictability (clearness)
    "interannual_cv_pct",     # year-to-year stability
    "solar_wind_monthly_corr",# hybrid hedging value
    "hot_days_per_year",      # PV heat-derating risk
]
X = df[FEATURES].copy()
X_scaled = StandardScaler().fit_transform(X)
pd.DataFrame(X_scaled, index=X.index, columns=X.columns).round(2)

# %% [markdown]
# ## 2. How many clusters? Hierarchical dendrogram + k-means diagnostics
#
# With only 13 sites, a dendrogram is more informative than committing to
# a k up front. It shows the actual merge structure so we can pick a
# defensible cut point, and we cross-check against k-means inertia/
# silhouette.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

Z = linkage(X_scaled, method="ward")
dendrogram(Z, labels=X.index.tolist(), ax=axes[0], leaf_rotation=90)
axes[0].set_title("Hierarchical clustering (Ward linkage)")
axes[0].set_ylabel("Distance")

inertias, silhouettes = [], []
k_range = range(2, 7)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

axes[1].plot(list(k_range), inertias, marker="o")
axes[1].set_title("K-means inertia (elbow)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Inertia")

axes[2].plot(list(k_range), silhouettes, marker="o", color="darkorange")
axes[2].set_title("K-means silhouette score")
axes[2].set_xlabel("k")
axes[2].set_ylabel("Silhouette")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cluster_selection_diagnostics.png", dpi=150)
plt.show()

print("Silhouette by k:", dict(zip(k_range, [round(s, 3) for s in silhouettes])))

# %% [markdown]
# Silhouette actually peaks at **k=2** (0.511) and decreases monotonically
# after that (0.452, 0.469, 0.423, 0.352 for k=3..6). k=2 is the
# "cleanest" split by that metric, but it just separates "coast/plateau"
# from "everything in or near the Sahara," which is the geographic
# gradient we already knew about from the EDA.
#
# **We choose k=4 instead**, accepting the small silhouette cost
# (0.469 vs. 0.511), because it's the smallest k that isolates two
# distinctions that matter for the siting question and that k=2/k=3
# collapse:
# - **Adrar splits off as its own singleton cluster.** Its extreme heat
#   exposure (87 hot-days/yr) makes it behave differently from the other
#   high-resource Saharan sites even though its solar/wind numbers are
#   similar to theirs.
# - **A distinct transition cluster (Djelfa, Laghouat, El Bayadh)**
#   separates from both the coastal/plateau group and the deep-Sahara
#   group, rather than being absorbed into one of them.
#
# This is a judgment call, not a mechanical "pick the max silhouette,"
# worth stating explicitly rather than quietly picking k=4 and only
# reporting the diagnostic that supports it.

# %%
K = 4

hier = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X_scaled)
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X_scaled)

df["cluster_hier"] = hier.labels_
df["cluster_kmeans"] = km.labels_

pd.crosstab(df["cluster_hier"], df["cluster_kmeans"])

# %% [markdown]
# ## 3. Do the clusters line up with the hand-labeled zones?

# %%
pd.crosstab(df["zone"], df["cluster_hier"])

# %% [markdown]
# They mostly do, with one specific mismatch worth calling out:
# **Djelfa** is hand-labeled "High Plateau" (like Batna and Tiaret), but
# it clusters with the Saharan Atlas sites (Laghouat, El Bayadh) instead.
# Its clearness index, interannual stability, and complementarity
# profile behave more like the Saharan-Atlas transition zone than like
# the other two Plateau sites. The hand-drawn zone boundary between
# "High Plateau" and "Saharan Atlas" doesn't line up with actual climate
# behavior at Djelfa specifically. Everything else falls in line with
# the zones we'd expect: Tell + rest of the Plateau together, and the
# four highest-resource Saharan sites (Ghardaïa, Ouargla, Béchar,
# Tamanrasset) together, with Adrar pulled out on its own by heat
# exposure alone.

# %% [markdown]
# ## 4. PCA projection: clusters vs. original zones, side by side

# %%
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, hue_col, title in zip(axes, ["zone", "cluster_hier"], ["Hand-labeled zone", "Data-driven cluster"]):
    sns.scatterplot(
        data=df, x="pc1", y="pc2", hue=hue_col, s=140, ax=ax, palette="colorblind"
    )
    for name, row in df.iterrows():
        ax.annotate(name, (row["pc1"], row["pc2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/pca_clusters_vs_zones.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Cluster profiles
#
# Mean of each raw (unscaled) feature per cluster: this is what actually
# gets described in the write-up, not the standardized values.

# %%
profile = df.groupby("cluster_hier")[FEATURES].mean().round(2)
profile["n_sites"] = df.groupby("cluster_hier").size()
profile["sites"] = df.groupby("cluster_hier").apply(lambda g: ", ".join(g.index), include_groups=False)
profile

# %% [markdown]
# ## Four renewable siting profiles
#
# **Cluster 0, Coast & Plateau, hybrid hedge** (Annaba, Algiers, Oran,
# Batna, Tiaret): moderate resource (GHI 4.94, kt 0.83), the weakest
# year-to-year stability of the four (CV 2.09%), but by far the
# strongest solar/wind hedge (corr −0.74). **Best case: hybrid solar+wind
# where the goal is reducing seasonal variability, not maximizing raw
# output.**
#
# **Cluster 1, Deep Sahara, high resource & aligned** (Ghardaïa,
# Ouargla, Béchar, Tamanrasset): strong resource (GHI 5.98, kt 0.94),
# very stable year to year (CV 0.92%), but solar and wind now move
# *together* (corr +0.50) and moderate heat exposure appears (~18
# hot-days/yr). **Best case: large solar-only builds. A hybrid plant
# here adds capacity, not risk reduction.**
#
# **Cluster 2, Saharan Atlas transition** (Djelfa, Laghouat, El
# Bayadh): resource and stability sit between clusters 0 and 1, complementarity
# is still moderately negative (−0.33, a partial hedge), and it
# actually has the **highest mean wind speed of any non-outlier cluster**
# (6.35 m/s at 50m, above even the deep-Sahara cluster's 5.89). That's a
# genuinely non-obvious result: on the wind axis alone, this
# "in-between" zone outperforms the flagship desert sites. **Best case:
# worth a dedicated wind-resource look, not just a leftover
# transition zone.**
#
# **Cluster 3, Adrar, heat-risk outlier** (singleton): the strongest raw
# resource of all 13 sites (GHI 6.08, kt 0.95, most stable at CV 0.88%)
# undercut by extreme heat exposure (87 hot-days/yr, 5x the next
# highest cluster). **Best case: excellent resource on paper, but not a
# safe pick without a real PV-derating and thermal-management analysis
# first.** Exactly the kind of site a resource map alone would rank #1
# and a siting study would catch before committing capital.

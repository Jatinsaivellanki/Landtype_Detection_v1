"""Run once:  python train.py --data data/EuroSAT_RGB --n 4000
Labels (folder names) are NEVER used for training; only for post-hoc checking."""
import argparse, glob, os, json, random
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score)
from features import extract, FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=4000)
args = ap.parse_args()
os.makedirs("artifacts", exist_ok=True); os.makedirs("figures", exist_ok=True)
SEED = 42; random.seed(SEED); rng = np.random.RandomState(SEED)

# ---------- 1. load + features ----------
paths = glob.glob(os.path.join(args.data, "*", "*.*"))
paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(paths); paths = paths[:args.n]
X = np.array([extract(Image.open(p)) for p in paths])
hidden = np.array([os.path.basename(os.path.dirname(p)) for p in paths])  # post-hoc only
df = pd.DataFrame(X, columns=FEATURE_NAMES)
print("Images:", len(df), "Features:", df.shape[1])

# ---------- 2. preprocessing ----------
df = df.replace([np.inf, -np.inf], np.nan)
df.isna().sum().to_csv("artifacts/missing_report.csv")
df.describe().T.to_csv("artifacts/feature_summary.csv")
imp = SimpleImputer(strategy="median"); sc = StandardScaler()
Xs = sc.fit_transform(imp.fit_transform(df))
# (no categorical fields: all features are numeric, label column is excluded)
plt.figure(figsize=(9, 7)); plt.imshow(np.corrcoef(Xs.T), cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=90, fontsize=6)
plt.yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, fontsize=6); plt.colorbar()
plt.title("Feature correlation (relevance / redundancy check)"); plt.tight_layout()
plt.savefig("figures/01_correlation.png", dpi=150); plt.close()

# ---------- 3. PCA ----------
full = PCA().fit(Xs); cum = np.cumsum(full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(range(1, len(cum) + 1), full.explained_variance_ratio_); ax[0].set_title("Scree")
ax[1].plot(range(1, len(cum) + 1), cum, "o-"); ax[1].axhline(.95, color="r", ls="--")
ax[1].set_title(f"Cumulative variance (95% at {n95} PCs)")
plt.tight_layout(); plt.savefig("figures/02_pca_variance.png", dpi=150); plt.close()
pca = PCA(n_components=n95, random_state=SEED).fit(Xs); Z = pca.transform(Xs)
pd.DataFrame(pca.components_[:3].T, index=FEATURE_NAMES, columns=["PC1", "PC2", "PC3"]).to_csv("artifacts/pca_loadings.csv")

# ---------- 4. choose k (metrics + stability) ----------
rows = []
for k in range(2, 11):
    labs = [KMeans(k, n_init=10, random_state=s).fit_predict(Z) for s in range(5)]
    ari = np.mean([adjusted_rand_score(labs[0], l) for l in labs[1:]])
    rows.append(dict(k=k, silhouette=silhouette_score(Z, labs[0], sample_size=min(2000, len(Z)), random_state=SEED),
                     davies_bouldin=davies_bouldin_score(Z, labs[0]),
                     calinski_harabasz=calinski_harabasz_score(Z, labs[0]), stability_ARI=ari))
met = pd.DataFrame(rows); met.to_csv("artifacts/k_selection.csv", index=False); print(met.round(3))
fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
for a, c in zip(ax, met.columns[1:]): a.plot(met.k, met[c], "o-"); a.set_title(c); a.set_xlabel("k")
plt.tight_layout(); plt.savefig("figures/03_k_selection.png", dpi=150); plt.close()
best_k = int(met[met.k >= 4].sort_values("silhouette", ascending=False).iloc[0].k)  # k>=4: land cover needs >3 groups
km = KMeans(best_k, n_init=20, random_state=SEED).fit(Z); labels = km.labels_
print("Chosen k =", best_k)

# ---------- 5. PCA 2D + t-SNE (perplexity comparison) ----------
idx = rng.choice(len(Z), min(3000, len(Z)), replace=False)
Z2 = Z[:, :2]
plt.figure(figsize=(6, 5)); plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=4, cmap="tab10")
plt.title("PCA 2D coloured by cluster"); plt.savefig("figures/04_pca2d.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5)); tsne_res = {}; tsne30 = None
for a, perp in zip(ax, [10, 30, 50]):
    t = TSNE(2, perplexity=min(perp, len(idx) - 2), init="pca", random_state=SEED)
    e = t.fit_transform(Z[idx]); tsne_res[perp] = float(t.kl_divergence_)
    a.scatter(e[:, 0], e[:, 1], c=labels[idx], s=4, cmap="tab10"); a.set_title(f"t-SNE perplexity={perp} (KL={t.kl_divergence_:.2f})")
    if perp == 30: tsne30 = e
plt.tight_layout(); plt.savefig("figures/05_tsne_perplexity.png", dpi=150); plt.close()
km_t = KMeans(best_k, n_init=10, random_state=SEED).fit(tsne30)
tsne_cmp = dict(silhouette_on_tsne=float(silhouette_score(tsne30, km_t.labels_)),
                ARI_pca_vs_tsne_clusters=float(adjusted_rand_score(labels[idx], km_t.labels_)))

# ---------- 6. profiles + post-hoc check ----------
prof = df.assign(cluster=labels).groupby("cluster").mean(); prof["size"] = np.bincount(labels); prof.to_csv("artifacts/cluster_profiles.csv")
ct = pd.crosstab(labels, hidden); ct.to_csv("artifacts/cluster_vs_hidden_labels.csv")
label_map = {int(c): dict(class_=str(ct.loc[c].idxmax()), purity=float(ct.loc[c].max() / ct.loc[c].sum())) for c in ct.index}
post = dict(ARI=float(adjusted_rand_score(hidden, labels)), NMI=float(normalized_mutual_info_score(hidden, labels)))
cm_scaled = np.array([Xs[labels == c].mean(0) for c in range(best_k)])
joblib.dump(dict(imp=imp, sc=sc, pca=pca, km=km, features=FEATURE_NAMES, k=best_k, cm_scaled=cm_scaled,
                 Z2=Z2.astype(np.float32), labels=labels, tsne_idx=idx, tsne_Z=Z[idx], tsne_emb=tsne30,
                 label_map=label_map, cumvar=cum), "artifacts/model.joblib")
json.dump(dict(n_images=len(df), n_features=len(FEATURE_NAMES), n_pc_95=n95, chosen_k=best_k, tsne_kl=tsne_res,
               tsne_vs_pca=tsne_cmp, posthoc=post, label_map=label_map), open("artifacts/results.json", "w"), indent=2)
print(json.dumps(post), "Done. Now run: streamlit run app.py")

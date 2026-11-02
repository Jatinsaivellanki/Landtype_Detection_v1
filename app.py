import numpy as np, pandas as pd, joblib, json
import streamlit as st, matplotlib.pyplot as plt
from PIL import Image
from features import extract

st.set_page_config(page_title="Land-Type Discovery", layout="wide")
M = joblib.load("artifacts/model.joblib")
st.title("🛰️ Unsupervised Land-Type Discovery (EuroSAT)")
st.caption("PCA + K-Means on interpretable patch features. Clusters are exploratory groups, NOT official land-cover classes.")

up = st.file_uploader("Upload a satellite patch (jpg/png)", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(up); x = extract(img).reshape(1, -1)
    xs = M["sc"].transform(M["imp"].transform(x)); z = M["pca"].transform(xs)
    d = M["km"].transform(z)[0]; c = int(d.argmin()); srt = np.sort(d)
    sep = (srt[1] - srt[0]) / srt[1]
    lm = M["label_map"][c]
    c1, c2 = st.columns([1, 2])
    c1.image(img.resize((256, 256)), caption="Input")
    c2.metric("Assigned cluster", f"Cluster {c}")
    c2.write(f"Closest known EuroSAT class (post-hoc only): **{lm['class_']}** — cluster purity {lm['purity']:.0%}")
    c2.write(f"Distance to centroid: **{d[c]:.2f}** | 2nd-nearest: {srt[1]:.2f} | Separation score: **{sep:.2f}** (0=ambiguous, 1=clear)")
    if sep < 0.15: c2.warning("Borderline sample: close to another cluster. Treat the assignment with caution.")

    st.subheader("1. Distance to every cluster centroid (PCA space)")
    st.bar_chart(pd.DataFrame({"distance": d}, index=[f"C{i}" for i in range(len(d))]))

    st.subheader("2. Features that drove the assignment")
    df = pd.DataFrame({"feature": M["features"], "raw value": x[0], "z-score": xs[0],
                       "cluster mean z": M["cm_scaled"][c]})
    df["gap"] = (df["z-score"] - df["cluster mean z"]).abs()
    df["match"] = np.where(df.gap < 0.75, "✅ typical", "⚠️ unusual")
    st.dataframe(df.sort_values("gap").round(3), use_container_width=True)

    st.subheader("3. Where it lands in the reduced space")
    a, b = st.columns(2)
    f, ax = plt.subplots(); ax.scatter(*M["Z2"].T, c=M["labels"], s=3, cmap="tab10", alpha=.5)
    ax.scatter(z[0, 0], z[0, 1], c="red", marker="*", s=300, edgecolor="k"); ax.set_title("PCA 2D (PC1 vs PC2)"); a.pyplot(f)
    nn = np.argsort(((M["tsne_Z"] - z) ** 2).sum(1))[:10]; p = M["tsne_emb"][nn].mean(0)
    f, ax = plt.subplots(); ax.scatter(*M["tsne_emb"].T, c=M["labels"][M["tsne_idx"]], s=3, cmap="tab10", alpha=.5)
    ax.scatter(*p, c="red", marker="*", s=300, edgecolor="k"); ax.set_title("t-SNE (approx. placement via 10 nearest neighbours)"); b.pyplot(f)
    st.caption("t-SNE has no transform for new points, so the star is an approximation.")

    st.subheader("4. Cluster profile (mean raw features of this cluster)")
    st.dataframe(pd.read_csv("artifacts/cluster_profiles.csv").set_index("cluster").loc[[c]].T.round(3))

with st.expander("Study metrics (PCA variance, k selection, t-SNE comparison, post-hoc check)"):
    st.json(json.load(open("artifacts/results.json")))
    st.dataframe(pd.read_csv("artifacts/k_selection.csv").round(3))
    for f in ["02_pca_variance", "03_k_selection", "05_tsne_perplexity"]: st.image(f"figures/{f}.png")
with st.expander("Limitations & responsible use"):
    st.markdown("- Hand-crafted RGB features only; no NIR/SWIR bands.\n- Clusters depend on k, features, seed and sample; not permanent categories.\n"
                "- EuroSAT is Europe-only (Sentinel-2): results may not transfer to other regions/seasons/sensors.\n"
                "- Hidden labels used only for post-hoc sanity checks, never for training.")

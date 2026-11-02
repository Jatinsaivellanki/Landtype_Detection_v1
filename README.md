# Unsupervised Land-Type Discovery (EuroSAT, labels hidden)
Features -> median impute -> StandardScaler -> PCA (95% variance) -> K-Means (k by silhouette/DB/CH/stability) -> t-SNE comparison -> Streamlit app.

## Prerequisites
Python 3.10+. Dataset: EuroSAT RGB (https://github.com/phelber/EuroSAT, direct zip: https://madm.dfki.de/files/sentinel/EuroSAT.zip). Unzip to `data/EuroSAT_RGB/` (10 class sub-folders; names are used only for post-hoc checks).

## Install & run
    pip install -r requirements.txt
    python train.py --data data/EuroSAT_RGB --n 4000   # creates artifacts/ and figures/
    streamlit run app.py
## Deploy
Push repo (including artifacts/ and figures/) to GitHub -> share.streamlit.io -> select app.py.

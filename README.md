# LandScope | Unsupervised Land-Type Discovery from Satellite Images

<p align="center">
  <strong>Discovering hidden visual patterns in satellite imagery using unsupervised machine learning.</strong>
</p>

<p align="center">
  <a href="https://landscopev1.streamlit.app/">Live Demo</a> •
  <a href="https://github.com/Jatinsaivellanki/Landtype_Detection_v1">GitHub Repository</a> •
  <a href="https://www.kaggle.com/code/jatin231226/notebooke546eecf3b">Kaggle Notebook</a> •
  <a href="https://www.kaggle.com/datasets/apollo2506/eurosat-dataset">Dataset</a>
</p>

---

## Overview

**LandScope** is an unsupervised machine learning project that explores how satellite image patches can be grouped according to their visual characteristics without providing predefined land-use labels during model training.

The project uses the EuroSAT satellite imagery dataset and extracts interpretable numerical features from individual image patches. These features are preprocessed, standardised and transformed using Principal Component Analysis (PCA) before being analysed through K-Means clustering.

The project also explores the resulting feature space using t-distributed Stochastic Neighbour Embedding (t-SNE), evaluates clustering configurations using multiple internal validation metrics, and investigates the visual characteristics of the discovered groups.

To make the results accessible, LandScope includes an interactive Streamlit application where users can upload a satellite image and explore its assigned cluster, distance from cluster centroids, feature-level characteristics and visual placement.

**The central objective is not to build a conventional land-cover classifier, but to investigate whether meaningful visual patterns can emerge from satellite imagery without supervised training labels.**

## Live Application

<https://landscopev1.streamlit.app/>

The application allows users to:

- Upload a satellite image patch in JPG, JPEG or PNG format.
- Extract numerical visual features from the uploaded image.
- Assign the image to its nearest discovered cluster.
- Examine distances from all cluster centroids.
- Inspect feature-level similarities and differences.
- Explore the image's position in PCA and approximate t-SNE visualisations.
- Examine the selected cluster's numerical profile.
- Review experimental metrics and model limitations.

## Problem Statement

Satellite imagery contains complex visual information, including variations in colour, vegetation, texture, water coverage and built-up structures.

Manually identifying recurring patterns across large collections of satellite images can be difficult.

Traditional supervised approaches require labelled examples to learn predefined categories. However, this project investigates a different question:

**Can a machine learning algorithm discover recurring visual patterns in satellite images without being given their correct land-use categories beforehand?**

LandScope addresses this question through feature engineering, dimensionality reduction, clustering and visual interpretation.

## Dataset

**EuroSAT — Sentinel-2 Satellite Image Dataset**

Dataset source: [EuroSAT Dataset on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

EuroSAT is a land-use and land-cover dataset containing satellite image patches derived from Sentinel-2 imagery.

The original dataset contains:
- 27,000 labelled satellite images.
- 10 land-use and land-cover categories.
- 13 Sentinel-2 spectral bands in its multispectral version.

This implementation uses the RGB image representation.

**Label handling:** The original class information is excluded from feature extraction, preprocessing and clustering. Folder labels are retained separately for optional post-hoc evaluation and sanity checks. They are not supplied to K-Means as training targets.

## Methodology

The project follows a structured unsupervised learning pipeline.

### 1. Image Processing and Feature Extraction

Instead of directly clustering raw images, LandScope converts each satellite image patch into a numerical feature vector.

Images are converted to RGB and resized to 64 × 64 pixels before feature extraction.

The implementation extracts 19 hand-crafted numerical features across the following groups:

| Feature family | What it measures |
|---|---|
| RGB statistics | Mean and standard deviation of red, green and blue channels |
| HSV statistics | Hue, saturation and value characteristics |
| Vegetation-related indices | Excess Green (ExG) and green-red index |
| Colour relationship | Green-blue index |
| Edge information | Mean and standard deviation of Sobel edge magnitude |
| Texture information | Laplacian variance and grayscale entropy |

These measurements provide a compact numerical representation of colour, vegetation-related characteristics and image texture.

### 2. Data Preprocessing

Before applying dimensionality reduction, the extracted feature matrix undergoes preprocessing.

- Non-finite values are converted into missing values.
- Missing values are handled using median imputation.
- Numerical features are standardised using `StandardScaler`.
- Feature correlation is examined to understand relationships and potential redundancy.

Standardisation ensures that features with larger numerical scales do not disproportionately influence distance-based algorithms.

### 3. Principal Component Analysis (PCA)

PCA is used to reduce the dimensionality of the standardised feature matrix.

It transforms the original features into principal components, which are new variables representing directions of maximum variance.

The implementation:
- Examines individual and cumulative explained variance.
- Determines the number of components required to retain at least 95% of the variance.
- Transforms the feature matrix into the reduced PCA representation.
- Saves PCA component loadings for interpretation.

This reduces dimensional complexity while retaining most of the variation present in the original measurements.

### 4. Selecting the Number of Clusters

LandScope evaluates multiple K-Means configurations rather than assuming a cluster count without experimental investigation.

The implementation tests K values from 2 to 10 and evaluates them using:

**Silhouette Score:** Measures how well observations fit within their assigned clusters compared with neighbouring clusters. Higher values are generally preferable.

**Davies–Bouldin Index:** Measures similarity between clusters using within-cluster dispersion and between-cluster separation. Lower values are generally preferable.

**Calinski–Harabasz Index:** Evaluates the ratio of between-cluster dispersion to within-cluster dispersion. Higher values generally indicate better-defined groups.

**Clustering Stability (ARI):** Compares cluster assignments across different initialisations using the Adjusted Rand Index.

The final K is selected using the implemented silhouette-based selection procedure, with K-Means fitted using multiple initialisations.

### 5. K-Means Clustering

K-Means partitions the PCA-transformed observations into groups based on their distances from cluster centroids.

The algorithm iteratively:
1. Initialises cluster centroids.
2. Assigns observations to their nearest centroid.
3. Updates centroid positions.
4. Repeats until convergence or the stopping condition is reached.

The resulting cluster assignments represent discovered patterns in the numerical feature space.

These groups are exploratory clusters and should not automatically be interpreted as official land-cover categories.

### 6. t-SNE Visual Exploration

t-SNE is used to explore the high-dimensional feature relationships in two dimensions.

The implementation compares different perplexity settings:

- Perplexity = 10
- Perplexity = 30
- Perplexity = 50

The resulting visualisations help investigate local neighbourhood structure and how observations appear in a two-dimensional representation.

The project also compares clustering behaviour between PCA and t-SNE representations.

**Important:** t-SNE is used for visual exploration and comparison. Its axes do not correspond to original physical features, and apparent visual separation does not independently establish cluster correctness.

### 7. Cluster Profiling and Post-hoc Evaluation

After clustering, the project generates numerical cluster profiles using the mean feature values of observations assigned to each cluster.

The original EuroSAT labels are optionally used after clustering for post-hoc comparison through:

- Adjusted Rand Index (ARI).
- Normalised Mutual Information (NMI).
- Cluster-to-class correspondence and cluster purity.

These measures help investigate how the discovered groups relate to the dataset's original categories.

They are evaluation tools, not supervised training objectives.

## Interactive Application

The Streamlit application extends the experimental workflow into an interactive image-analysis interface.

### Image Upload and Cluster Assignment

Users upload a satellite patch, which is processed using the same feature extraction and preprocessing objects saved during model training.

The application then:
- Applies median imputation and standardisation.
- Transforms the feature vector using the fitted PCA model.
- Calculates distances to the K-Means centroids.
- Assigns the image to its nearest cluster.

### Cluster Distance and Assignment Confidence

The application displays:
- Assigned cluster.
- Distance to the nearest centroid.
- Distance to the second-nearest centroid.
- A separation score based on the difference between these distances.

A low separation score indicates that the sample is relatively close to another cluster and should be interpreted cautiously.

This is a distance-based separation indicator, not a calibrated probability of correctness.

### Feature-Level Interpretation

The application compares the uploaded image's standardised feature values with the mean standardised feature values of the assigned cluster.

It highlights features that are relatively typical or unusual compared with the cluster profile.

This provides a more interpretable view of why an image may fit a particular group.

### Visual Exploration

The application displays the uploaded image against:
- A PCA two-dimensional projection.
- An approximate t-SNE placement based on its nearest neighbours.

Because standard t-SNE does not provide a general-purpose transform for unseen samples, the application's t-SNE position is an approximation rather than a direct embedding.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| SciPy | Image-processing operations |
| Pillow | Image loading and processing |
| Scikit-learn | Preprocessing, PCA, t-SNE, K-Means and evaluation |
| Matplotlib | Data visualisation |
| Joblib | Saving and loading trained model objects |
| Streamlit | Interactive web application |

## Repository Structure

```text
Landtype_Detection_v1/
│
├── artifacts/
│   ├── model.joblib
│   ├── cluster_profiles.csv
│   ├── k_selection.csv
│   ├── results.json
│   └── ...
│
├── data/
│   └── EuroSAT_RGB/
│       ├── AnnualCrop/
│       ├── Forest/
│       ├── HerbaceousVegetation/
│       └── ...
│
├── figures/
│   ├── 01_correlation.png
│   ├── 02_pca_variance.png
│   ├── 03_k_selection.png
│   ├── 04_pca2d.png
│   └── 05_tsne_perplexity.png
│
├── app.py
├── features.py
├── train.py
├── requirements.txt
└── README.md
```

**Main files:**

- `features.py` — Implements image preprocessing and hand-crafted feature extraction.
- `train.py` — Executes the experimental pipeline, dimensionality reduction, cluster selection, clustering, evaluation and artifact generation.
- `app.py` — Implements the interactive Streamlit application.
- `requirements.txt` — Lists the Python dependencies.
- `artifacts/` — Stores trained models, numerical results and cluster profiles.
- `figures/` — Stores generated experimental visualisations.

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- pip.
- EuroSAT RGB dataset.

### 1. Clone the Repository

```bash
git clone https://github.com/Jatinsaivellanki/Landtype_Detection_v1.git

cd Landtype_Detection_v1
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the EuroSAT RGB dataset from:

https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

Extract the dataset into:

```text
data/EuroSAT_RGB/
```

The expected structure contains the image-category subdirectories inside `EuroSAT_RGB`.

### 5. Train the Model

Run:

```bash
python train.py --data data/EuroSAT_RGB --n 4000
```

The `--n` parameter specifies the maximum number of image patches to process.

The training script generates experimental artifacts and visualisations in the `artifacts/` and `figures/` directories.

### 6. Launch the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access LandScope.

## Reproducibility

The training script uses a fixed random seed of 42 for its principal random operations and model initialisations.

The trained preprocessing objects, PCA transformation and K-Means model are saved using Joblib.

The application loads these saved objects to process new image uploads without retraining the model.

For reproducible results, use the same dataset version, feature extraction implementation, dependencies and training configuration.

## Limitations

LandScope is an exploratory unsupervised learning system. Its results should be interpreted within the following limitations:

- **Hand-crafted features:** The implementation uses RGB-derived measurements rather than the complete multispectral information available from Sentinel-2.
- **Feature representation:** Numerical summaries cannot capture every spatial and semantic characteristic of satellite imagery.
- **Cluster interpretation:** Discovered clusters do not necessarily correspond to official land-use categories.
- **Parameter sensitivity:** Results may change with the selected K, random initialisation, feature representation and sample selection.
- **Dimensionality reduction:** PCA and t-SNE provide different representations and may not preserve every relationship in the original feature space.
- **Geographical generalisation:** EuroSAT-based findings should not automatically be generalised to different geographical regions, seasons or satellite sensors.
- **Post-hoc evaluation:** Agreement with original dataset labels does not turn an unsupervised clustering result into a supervised classifier.

## Future Improvements

Potential directions for extending LandScope include:

- Incorporating Sentinel-2 multispectral bands, including NIR and SWIR information.
- Exploring deep learning-based image embeddings.
- Comparing additional clustering algorithms such as DBSCAN, Agglomerative Clustering and Gaussian Mixture Models.
- Investigating spatially aware clustering methods.
- Improving cluster interpretability through feature importance and visual explanations.
- Evaluating robustness across different geographical regions and datasets.

## Project Resources

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/Jatinsaivellanki/Landtype_Detection_v1 |
| Live Streamlit Application | https://landscopev1.streamlit.app/ |
| Kaggle Notebook | https://www.kaggle.com/code/jatin231226/notebooke546eecf3b |
| EuroSAT Dataset | https://www.kaggle.com/datasets/apollo2506/eurosat-dataset |

## Acknowledgements

This project uses the EuroSAT satellite imagery dataset for exploring unsupervised land-pattern discovery.

We acknowledge the dataset contributors and the open-source Python ecosystem, particularly the developers of NumPy, Pandas, SciPy, Pillow, Scikit-learn, Matplotlib and Streamlit.

---

**LandScope — Turning satellite image patches into measurable patterns and interpretable clusters.**

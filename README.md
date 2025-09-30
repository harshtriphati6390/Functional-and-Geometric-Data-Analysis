# Functional-and-Geometric-Data-Analysis
Methods for analyzing functions, curves, shapes; treating data in spatial/temporal continuous spaces, multiscale geometry and topological data analysis. 
Step-by-step project plan
<img width="334" height="418" alt="Screenshot 2025-09-30 234044" src="https://github.com/user-attachments/assets/8323a7ed-cdae-4c50-90d5-be1f42f43cc9" />


<img width="696" height="418" alt="Screenshot 2025-09-30 233912" src="https://github.com/user-attachments/assets/cd5768d0-3cfd-4604-903f-23788ec1a0ac" />

<img width="662" height="427" alt="Screenshot 2025-09-30 234017" src="https://github.com/user-attachments/assets/e8e7d607-2480-4e0e-aa27-668f759eaa28" />




1) Define the question & success criteria
Tools & libraries (quick list)

Python: numpy, pandas, scipy, scikit-learn, matplotlib, skfda (for more FDA utilities), geomstats (manifold stats), ripser + persim (TDA).

R: fda, fdapace (FPCA/PACE), shapes (Procrustes / shape analysis).

Visualization: matplotlib, plotly, seaborn.

Dashboard: Streamlit, Voila.
0) Quick project setup

Goal: set up environment and folder structure.
Tools: Python 3.9+ (conda or venv), Jupyter notebook or VS Code. Libraries: numpy, pandas, scipy, matplotlib, scikit-learn, scikit-fda (optional), geomstats (optional), ripser, persim
Actions:

# (conda)
conda create -n fgda python=3.9 -y
conda activate fgda
pip install numpy pandas scipy matplotlib scikit-learn scikit-fda geomstats ripser persim jupyter


Expected: environment with necessary libs.
Tip: keep a data/, notebooks/, scripts/, results/ folder.
<img width="641" height="423" alt="Screenshot 2025-09-30 233938" src="https://github.com/user-attachments/assets/76f18d9a-abb9-4d0b-a3e7-c821df42c3ba" />
<img width="614" height="418" alt="Screenshot 2025-09-30 233952" src="https://github.com/user-attachments/assets/0f353bce-7bdf-47fd-8191-2de0c7c78f39" />

1) Define the question and success metric (1–2 lines)

Goal: make the objective concrete.
Actions (write it down):

e.g., “Classify gait as normal vs abnormal. Metric = accuracy and AUC.”
Tip: choose primary metric and baseline model (e.g., logistic regression).

2) Collect or prepare data

Goal: gather functional data (curves) and geometric data (landmarks or silhouettes).
Tools: any data source; if images, opencv or scikit-image for extraction.
Actions:

Place files in data/functional/ (e.g. .csv or .npy) and data/geometric/ (images or landmarks/).

Load in Python:

import numpy as np, pandas as pd
X_func = np.load("data/functional/X_func.npy")   # shape (n_samples, n_timepoints)
times = np.load("data/functional/times.npy")     # time grid
# landmarks: list of arrays, each shape (n_landmarks, 2)
import glob
landmarks = [np.load(f) for f in sorted(glob.glob("data/geometric/landmarks_*.npy"))]

<img width="606" height="436" alt="Screenshot 2025-09-30 234032" src="https://github.com/user-attachments/assets/120de31e-554b-4631-97f5-61065770eb97" />

Checks: shapes, missing values, time grid regularity.
Pitfall: irregular time sampling — keep raw times if irregular.

3) Exploratory Data Analysis (EDA) — visualize raw data

Goal: understand n<img width="606" height="436" alt="Screenshot 2025-09-30 234032" src="https://github.com/user-attachments/assets/65f8095a-0cd3-4616-b313-8cd2b899d5ab" />
oise, phase shifts, and shape variability.
Tools: matplotlib, seaborn (optional)
Actions:

import matplotlib.pyplot as plt
# overlay a few curves
for i in range(10):
    plt.plot(times, X_func[i], alpha=0.6)
plt.title("Sample functional curves")
# visualize a shape
plt.scatter(landmarks[0][:,0], landmarks[0][:,1])
plt.gca().invert_yaxis()  # if image-based coords


Checks: are peaks at same times? are shapes centered?
Tip: do quick statistics: mean curve, std band.

4) Functional preprocessing — smoothing & basis representation

Goal: denoise and convert discrete curves into smooth functional objects (coefficients).
Tools: scipy.interpolate.UnivariateSpline, skfda (for higher-level FDA) or basis via scipy/numpy.
Actions: smoothing + basis coefficients:

from scipy.interpolate import UnivariateSpline
alpha = 0.5  # smoothing parameter — tune!
X_smooth = np.array([UnivariateSpline(times, y, s=alpha)(times) for y in X_func])


Or convert to B-spline basis (skfda example):

from skfda.representation.basis import BSplineBasis
from skfda.representation import FDataGrid
fd = FDataGrid(data_matrix=X_func, grid_points=times)
basis = BSplineBasis(n_basis=15)
fd_basis = fd.to_basis(basis)
coeffs = fd_basis.coefficients  # n_samples x n_basis


Expected: X_smooth (n × t) and/or coeffs (n × nbasis).
Checks: plot smoothed curves vs raw; tune alpha or n_basis with visual inspection / CV.

5) Functional alignment / registration (remove phase variability)

Goal: align peaks/features across samples so amplitude variation is meaningful.
Tools: tslearn or dtaidistance for DTW; skfda.registration for registration in R/Python.
Actions: simple DTW alignment (python example using tslearn):

from tslearn.metrics import dtw_path
# align sample i to a template (mean curve)
template = X_smooth.mean(axis=0)
path, dist = dtw_path(template, X_smooth[0])
# use path to warp X_smooth[0] to template — or use built-in registration funcs


Expected: aligned curves with synchronized peaks.
Checks: overlay mean and sample before/after alignment.
Tip: registration changes timing information — only do if timing differences are nuisance.

6) Functional dimension reduction — FPCA

Goal: reduce each function to a few scores (features).
Tools: skfda FPCA or manual SVD on discretized grid.
Actions (SVD quick method):

Xc = X_smooth - X_smooth.mean(axis=0)   # center
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
k = 5
fpca_scores = U[:, :k] * S[:k]           # n_samples x k
eigenfunctions = Vt[:k, :]               # k x n_timepoints


Expected: fpca_scores (compact numeric features).
Checks: fraction of variance explained by first k components; reconstruction error.

7) Geometric preprocessing — landmarks & Procrustes alignment

Goal: make shapes comparable by removing translation, rotation, and scale.
Tools: scipy.spatial.procrustes, geomstats (for manifold-aware methods).
Actions:

from scipy.spatial import procrustes
reference = landmarks[0]
aligned = []
for L in landmarks:
    mtx1, mtx2, disparity = procrustes(reference, L)
    aligned.append(mtx2.flatten())   # flatten to vector
aligned = np.array(aligned)  # shape (n_samples, n_landmarks*2)


Expected: aligned matrix for downstream analysis.
Checks: plot a few aligned shapes to verify orientation/scale removed.

8) Geometric dimension reduction / manifold features

Goal: get low-dim coordinates capturing nonlinear shape variation.
Tools: sklearn.manifold (Isomap, MDS, SpectralEmbedding), geomstats for tangent PCA.
Actions (Isomap example):

from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=10, n_components=3)
shape_emb = iso.fit_transform(aligned)  # n_samples x 3


Or tangent-space PCA (geomstats) if shapes lie on a specific manifold — use geomstats docs.
Expected: shape_emb features.
Checks: scatter-plot embedding colored by labels; try different n_neighbors.

9) Topological features (optional but powerful for shapes/point clouds)

Goal: capture holes/loops and robust shape descriptors using persistence.
Tools: ripser, persim (python).
Actions:

from ripser import ripser
from persim import persim
dgm = ripser(point_cloud)['dgms']  # list of diagrams by dimension
# convert to simple numeric summaries: total lifetime, number of long bars, etc.


Expected: vector of TDA summaries per sample.
Checks: compare persistence diagrams for two example shapes; use stability under noise.

10) Form the feature table (concatenate features)

Goal: combine FPCA scores + shape embeddings + TDA summaries + any scalar covariates.
Actions:

import numpy as np
features = np.hstack([fpca_scores, shape_emb, tda_summary_array])
# standardize:
from sklearn.preprocessing import StandardScaler
features = StandardScaler().fit_transform(features)


Expected: features shape (n_samples, p).
Checks: correlate features to see redundancy; use PCA to inspect overall variance.

11) Modeling (classification / regression / clustering)

Goal: fit predictive or descriptive models on combined features.
Tools: sklearn classifiers / regressors, e.g., RandomForestClassifier, LogisticRegression, XGBoost if desired.
Actions (example classification with RF):

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, features, labels, cv=cv, scoring='roc_auc')
print("AUC (cv):", scores.mean())


Expected: CV metric (AUC, accuracy).
Checks: confusion matrix, ROC curve, feature importances.
Tip: run ablation: (functional only) vs (geometric only) vs (combined).

12) Interpretation — map back to function/shape space

Goal: explain what model uses — show how principal modes change curves/shapes.
Actions:

For a given FPCA component phi, show mean ± c * phi plotted as curves to visualize mode of variation.

For shape PCA directions, convert PC vector back to landmark coordinates and plot ± directions.
Example (FPCA reconstruction):

mean_curve = X_smooth.mean(axis=0)
mode1 = eigenfunctions[0]
plt.plot(times, mean_curve, label='mean')
plt.plot(times, mean_curve + 2*mode1, label='+2 mode1')
plt.plot(times, mean_curve - 2*mode1, label='-2 mode1')
plt.legend()


Expected: figures that justify what features mean.

13) Validation, robustness & tests

Goal: ensure results aren’t brittle.
Actions / checks:

CV and nested CV for hyperparameter tuning.

Permutation test: shuffle labels to get null distribution of metric.

Stability: subsample data and re-run FPCA / embeddings to check stability.

Reconstruction error thresholds to pick number of FPCA components.
Tip: record seeds and versions for reproducibility.

14) Visualization & deliverables
    
Goal: package results for others.
Deliverables:

Jupyter notebook with pipeline, comments, and plots.

results/ with figures: mean curves, mode perturbations, shape galleries, embeddings, persistence diagrams, performance table.

Short report (1–2 pages) and README with requirements.txt.
Optional: interactive Streamlit app to explore samples and sliders for FPCA modes.

15) Common pitfalls & debugging tips

Over-smoothing: heavy smoothing removes real signal. Always visualize.

Misregistration misuse: only register if timing is nuisance; registration can remove meaningful timing differences.

Flattening manifold data: flattening shapes naively can lose geometry — use tangent-space PCA or manifold methods if curvature matters.

Small sample size: manifold / TDA estimators can be noisy for small n — report uncertainty.

Feature leakage: when building features, avoid using label information in preprocessing steps across train/test split — e.g. compute FPCA on training set only when doing CV.




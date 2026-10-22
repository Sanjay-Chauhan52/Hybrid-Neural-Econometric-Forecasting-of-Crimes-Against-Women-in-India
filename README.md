# Hybrid Neural-Econometric Forecasting of Crimes Against Women in India

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Backend-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/ML%2FDL-TensorFlow%20%7C%20Keras-orange.svg)](https://tensorflow.org/)
[![Frontend](https://img.shields.io/badge/Visualization-Plotly%20%7C%20Leaflet.js-green.svg)](https://plotly.com/javascript/)
[![Dataset](https://img.shields.io/badge/Dataset-NCRB%20(2001--2021)-red.svg)](https://www.kaggle.com/datasets/balajivaraprasad/crimes-against-women-in-india-2001-2021/data)

An end-to-end, multi-model spatio-temporal crime analytics and predictive decision-support system analyzing reported crimes against women across 36 Indian States and Union Territories over a 21-year period (2001–2021).

The platform bridges the gap between econometric time-series models and deep recurrent representation learning, featuring a **Tri-Signal PCA Crime Risk Index (CRI)** and an **Adaptive Inverse-Backtest-Error Forecast Ensemble**.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Project Objectives](#-project-objectives)
- [System Architecture](#-system-architecture)
- [Core Analytical Modules](#-core-analytical-modules)
  - [1. Tri-Signal PCA Crime Risk Index (CRI)](#1-tri-signal-pca-crime-risk-index-cri)
  - [2. Multi-Model Forecasting Engine](#2-multi-model-forecasting-engine)
- [Empirical Backtest Results](#-empirical-backtest-results)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Installation and Setup Guide](#-installation-and-setup-guide)
- [API Reference](#-api-reference)
- [References](#-references)

---

## 📖 Executive Overview

Analyzing crime against women in India presents three major challenges:
1. **Severe Category Scale Differences:** Annual domestic violence cases reach hundreds of thousands, while dowry deaths or human trafficking counts may be dozens to hundreds per state. Simple arithmetic aggregation causes low-volume, high-severity crimes to be overshadowed.
2. **Multivariate Distortion:** States have distinct profiles (e.g., disproportionately elevated kidnapping vs. domestic violence). Traditional indices assume category independence and fail to capture multivariate covariance.
3. **Forecasting Short Annual Time Series:** With only 21 annual observations (2001–2021), pure deep learning architectures tend to overfit and drift over multi-step horizons, whereas pure linear models cannot capture non-linear inertia or turning points.

**Our Solution:**
* **Unsupervised Risk Assessment:** A Tri-Signal Crime Risk Index fusing heuristic rankings, deep autoencoder reconstruction error, and Mahalanobis statistical distance via Principal Component Analysis (PCA).
* **Neural-Econometric Forecasting:** A hybrid ensemble coupling an econometric Damped Growth model with a Global Recurrent Neural Network (LSTM with 4D State Entity Embeddings), weighted dynamically by inverse historical backtest error.

---

## 🎯 Project Objectives

1. **Standardized Panel Preprocessing:** Curate and validate 736 State–Year observations from the National Crime Records Bureau (NCRB) across 36 States/UTs.
2. **Multi-Signal Risk Indexing:** Move beyond raw counts to an un-dominated 0–100 Crime Risk Index (CRI) with risk tertile classifications (*Low, Medium, High*).
3. **Interactive Spatio-Temporal Discovery:** Provide interactive choropleth maps, longitudinal heatmaps, temporal growth trajectories, and cross-state ranking tables.
4. **Multi-Model Competitive Forecasting:** Benchmark econometric baseline growth against state-embedded deep LSTM neural networks.
5. **Adaptive Ensembling & Uncertainty Quantification:** Generate optimal multi-year point forecasts with calibrated 90% confidence bands and comprehensive holdout backtesting (MAE, WAPE, Bias, Directional Accuracy).

---

## 🏗 System Architecture

```text
                               +------------------------------------------+
                               |     NCRB Dataset (2001-2021 Panel)       |
                               |    36 States/UTs • 7 Crime Categories    |
                               +--------------------+---------------------+
                                                    |
                                                    v
                               +--------------------+---------------------+
                               |   Data Preprocessing & Panel Normalizer   |
                               |    • Category Scaling • Zero Imputation   |
                               +--------------------+---------------------+
                                                    |
                   +--------------------------------+--------------------------------+
                   |                                                                 |
                   v                                                                 v
+------------------+------------------+                           +------------------+------------------+
|     Tri-Signal PCA Risk Engine      |                           |   Hybrid Multi-Model Forecaster     |
|                                     |                           |                                     |
|  [S1] Equal-Weight Percentile Rank  |                           |  [Model 1] Damped Growth Baseline   |
|  [S2] Autoencoder Reconstruction    |                           |            • Rolling Backtest Calib |
|  [S3] Mahalanobis Covariance Dist   |                           |  [Model 2] Global State-Embedded    |
|                 │                   |                           |            LSTM Neural Network      |
|                 v                   |                           |            • 4D Entity Embeddings   |
|    Principal Component Analysis     |                           |                 │                   |
|       (PC1 Fusion: 67.8% Var)       |                           |                 v                   |
|                 │                   |                           |  [Model 3] Inverse-Error Ensemble   |
|                 v                   |                           |            • Dynamic Horizon Weight |
|    Consensus CRI (0-100 Score)      |                           |            • 90% Confidence Bounds  |
+------------------+------------------+                           +------------------+------------------+
                   |                                                                 |
                   +--------------------------------+--------------------------------+
                                                    |
                                                    v
                               +--------------------+---------------------+
                               |         Flask Backend REST API           |
                               |   (/api/overview, /api/cri, /api/fc)     |
                               +--------------------+---------------------+
                                                    |
                                                    v
                               +--------------------+---------------------+
                               |   Interactive Web Dashboard (UI)        |
                               |   • Leaflet Maps  • Plotly Trajectories  |
                               |   • Risk Heatmaps • Model Comparison     |
                               +------------------------------------------+
```

---

## 🔬 Core Analytical Modules

### 1. Tri-Signal PCA Crime Risk Index (CRI)

The framework evaluates state-level crime risk through three complementary mathematical perspectives:

$$\mathbf{S} = [S_1,\; S_2,\; S_3]^T$$

1. **Signal 1 — Heuristic Percentile Rank ($S_1$):**
   Equal-weighted cross-sectional percentile across the 7 normalized crime categories.
   $$S_1(s, y) = \frac{1}{K} \sum_{k=1}^K P(s, k, y)$$
2. **Signal 2 — Deep Autoencoder Anomaly Signal ($S_2$):**
   An unsupervised symmetric neural network ($\mathbb{R}^7 \rightarrow \mathbb{R}^4 \rightarrow \mathbb{R}^7$) trained on reconstruction loss. Higher reconstruction MSE flags states with atypical or distorted crime compositions.
   $$S_2(s, y) = \text{Percentile}\left(\frac{1}{K} \sum_{k=1}^K (x_k - \hat{x}_k)^2\right)$$
3. **Signal 3 — Mahalanobis Statistical Distance ($S_3$):**
   Quantifies multivariate dispersion from the national centroid using the full regularized covariance matrix $\boldsymbol{\Sigma}_y$, accounting for collinearity among crimes:
   $$D_M(s, y) = \sqrt{(\mathbf{x}_{s,y} - \boldsymbol{\mu}_y)^T (\boldsymbol{\Sigma}_y + \epsilon \mathbf{I})^{-1} (\mathbf{x}_{s,y} - \boldsymbol{\mu}_y)}$$
4. **PCA Consensus Fusion:**
   Standardizing $\mathbf{S}$ and projecting onto the first principal component loading vector $\mathbf{v}_1$:
   $$\text{CRI}_{\text{raw}} = 0.4912 \cdot Z_{S_1} + 0.5869 \cdot Z_{S_2} + 0.6437 \cdot Z_{S_3}$$
   $$\text{CRI}(s, y) = \text{Percentile}(\text{CRI}_{\text{raw}}) \in [0, 100]$$

---

### 2. Multi-Model Forecasting Engine

The platform provides three competitive forecasting approaches:

#### Model 1: Econometric Damped Growth (Baseline)
Uses a recent 3-observation baseline $B_0 = \frac{1}{3}\sum_{i=0}^2 C_{Y_0-i}$ and compounds mean historical YoY growth $g$, dampened exponentially across forecast horizon $h$ by parameter $d \in [0.05, 0.95]$ selected via rolling backtests:
$$g_h^* = \text{clip}(g \cdot d^{h-1},\; -0.75,\; 0.75)$$
$$\hat{C}_{\text{growth}, Y_0+h} = \hat{C}_{\text{growth}, Y_0+h-1} \cdot (1 + g_h^*)$$

#### Model 2: State-Embedded LSTM (Deep Learning)
* **Single Global Network:** Avoids sample-scarcity by training across all 36 states simultaneously.
* **4D Entity Embedding:** A learned lookup matrix $\mathbf{E} \in \mathbb{R}^{36 \times 4}$ provides spatial context and individual state baseline scaling.
* **Architecture:** Sequence Input $(3 \times 1) \rightarrow \text{LSTM}(16) \rightarrow \text{Dropout}(0.2) \oplus \mathbf{e}_s \rightarrow \text{Dense}(8, \text{ReLU}) \rightarrow \text{Dense}(1, \text{Linear})$.
* Trained with Huber loss ($\delta = 0.3$) and Adam optimizer. Inverts log-minmax normalization autoregressively.

#### Model 3: Adaptive Inverse-Backtest-Error Ensemble
Blends deep learning representation with econometric baseline stability using inverse holdout backtest error weights:


$$
w_{\text{lstm}}(h) =
\frac{\frac{1}{\text{MAE}{\text{lstm}}(h)}}
{\frac{1}{\text{MAE}{\text{lstm}}(h)} +
\frac{1}{\text{MAE}_{\text{growth}}(h)}}
$$

The growth-model weight is:

$$
w_{\text{growth}}(h) = 1 - w_{\text{lstm}}(h)
$$

The final ensemble forecast is computed as:



$$
\hat{C}{\text{ens},Y_0+h} =
w{\text{lstm}}(h)\hat{C}{\text{lstm},Y_0+h}
+
w{\text{growth}}(h)\hat{C}_{\text{growth},Y_0+h}
$$



---

## 📊 Empirical Backtest Results

### National Multi-Horizon Backtest Performance (Clean States)

| Horizon Step ($h$) | Model | Backtest MAE (Cases) | Median MAPE (%) | Horizon Ensemble Weight ($w$) |
| :---: | :--- | :---: | :---: | :---: |
| **$h = 1$ Year (2022)** | Damped Growth Baseline | 1,335.7 | 15.9% | $0.5417$ |
| | LSTM Neural Network | 1,578.9 | 18.4% | $0.4583$ |
| | **Ensemble (Hybrid)** | **800.4** | **16.6%** | — |
| **$h = 2$ Years (2023)** | Damped Growth Baseline | 6,763.7 | 56.0% | $0.4834$ |
| | LSTM Neural Network | 6,328.1 | 51.8% | $0.5166$ |
| | **Ensemble (Hybrid)** | **6,345.1** | **48.8%** | — |
| **$h = 3$ Years (2024)** | Damped Growth Baseline | 7,618.3 | 55.6% | $0.5042$ |
| | LSTM Neural Network | 7,748.4 | 58.7% | $0.4958$ |
| | **Ensemble (Hybrid)** | **7,402.4** | **57.1%** | — |

> **Key Result:** At the 1-year lookahead ($h=1$), the Ensemble achieves an MAE of **800.4 cases**, representing an error reduction of **$40.1\%$** over baseline growth and **$49.3\%$** over standalone LSTM.

---

## 📁 Dataset Description

* **Source:** National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India.
* **Panel Dimensions:** 736 State–Year records across 36 States & Union Territories.
* **Temporal Coverage:** 2001–2021 (21 annual intervals).
* **Monitored Crime Categories ($K=7$):**
  1. Rape
  2. Kidnapping & Abduction (`K&A`)
  3. Dowry Deaths (`DD`)
  4. Assault against Women / Molestation (`AoW`)
  5. Assault against Modesty of Women / Harassment (`AoM`)
  6. Domestic Violence / Cruelty by Husband or Relatives (`DV`)
  7. Women Trafficking (`WT`)
  * Aggregate: `Total Crime`

---

## 📂 Project Structure

```text
├── backend/
│   ├── analytics.py             # Spatio-temporal metrics, CRI heuristics, baseline growth
│   ├── analytics_ml.py          # ML pipelines: LSTM, Autoencoder, Mahalanobis, PCA, Ensembles
│   ├── app.py                   # Flask web server & REST API endpoints
│   └── ml_data_utils.py         # Anomaly detection & data repairing utilities
├── dataset/
│   └── cleaned_crime_data.csv   # Standardized NCRB crime panel
├── frontend/
│   ├── css/
│   │   └── style.css            # Responsive UI styles and theme definitions
│   ├── index.html               # Multi-tab single page application
│   └── js/
│       └── app.js               # Plotly/Leaflet rendering, reactive state management
├── models/
│   ├── cri_autoencoder.keras    # Trained Keras Autoencoder weights
│   ├── cri_ensemble_meta.json   # PCA projection vector and validation stats
│   ├── forecast_lstm.keras      # Trained global State-Embedded LSTM network
│   └── forecast_ensemble_meta.json # Normalization parameters & backtest weights
├── script/
│   ├── train_cri_ensemble.py    # Training pipeline for CRI Autoencoder and PCA
│   └── train_forecast_ensemble.py # Training pipeline for Global LSTM & backtest weights
├── requirements.txt             # Python environment dependencies
└── run.py                       # Main application runner
```

---

## 🚀 Installation and Setup Guide

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* Modern web browser (Chrome, Firefox, Edge, Safari)
* Git installed

### 2. Clone the Repository
```bash
git clone https://github.com/Sanjay-Chauhan52/Hybrid-Neural-Econometric-Forecasting-of-Crimes-Against-Women-in-India.git
cd Hybrid-Neural-Econometric-Forecasting-of-Crimes-Against-Women-in-India
```

### 3. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python run.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 API Reference

| Endpoint | Method | Parameters | Description |
| :--- | :---: | :--- | :--- |
| `/api/meta` | `GET` | — | Returns dataset metadata, states list, years, and crime categories. |
| `/api/overview` | `GET` | `year`, `crime` | National and state summary statistics and YoY growth. |
| `/api/map` | `GET` | `year`, `metric`, `crime` | State-wise metrics for choropleth rendering. |
| `/api/cri` | `GET` | `year` | Ranked CRI scores, component values, and risk categories. |
| `/api/forecast` | `GET` | `state`, `crime`, `horizon`, `origin_year` | Baseline econometric growth forecast. |
| `/api/forecast-ml` | `GET` | `state`, `model`, `horizon`, `origin_year` | Evaluates selected ML model (`growth`, `lstm`, `ensemble`). |
| `/api/forecast-compare-ml` | `GET` | `state`, `horizon`, `origin_year` | Returns comparative multi-model predictions and backtest holdouts. |

---

## 📚 References

1. **Dwivedi, N., & Sachdeva, S.** (2019). Gender-based violence in New Delhi, India: Forecast based on secondary data analysis. *Eastern Mediterranean Health Journal*, 25(4), 262–268.
2. **Bhardwaj, G., & Bawa, R. K.** (2022). Assaying the Statistics of Crime Against Women in India using Provenance and Machine Learning Models. *International Journal of Advanced Computer Science and Applications (IJACSA)*, 13(7), 492–501.
3. **Gupta, U., & Sharma, R.** (2023). Analysis of criminal spatial events in India using exploratory data analysis and regression. *Computers and Electrical Engineering*, 109(Part A), 108761.
4. **National Crime Records Bureau (NCRB)**, Ministry of Home Affairs, Government of India. *Crime in India Annual Reports (2001–2021)*.

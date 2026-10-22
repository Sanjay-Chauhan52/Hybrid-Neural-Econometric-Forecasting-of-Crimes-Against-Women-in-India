# Hybrid Neural-Econometric Forecasting of Crimes Against Women in India

An integrated, multi-model spatio-temporal crime analytics framework for reported crimes against women across 36 Indian States and Union Territories (2001–2021).

## Key Features

1. **Tri-Signal PCA Crime Risk Index (CRI):**
   - Composite risk index fusing heuristic percentile rankings ($S_1$), deep unsupervised Autoencoder anomaly reconstruction error ($S_2$), and Mahalanobis multivariate covariance statistical distance ($S_3$) via Principal Component Analysis (explaining $67.77\%$ shared variance).
2. **Hybrid Multi-Model Forecasting Engine:**
   - **Econometric Damped Growth Baseline:** compounds historical mean YoY growth with rolling-origin calibrated damping.
   - **Deep LSTM Neural Network:** global recurrent neural network with learned 4-dimensional State Entity Embeddings.
   - **Adaptive Ensemble:** blends deep learning flexibility with econometric stability via inverse-backtest-error weighting, accompanied by empirical 90% confidence bands.
3. **Interactive Decision-Support Dashboard:**
   - Spatial choropleth mapping, 21-year State–Year heatmaps, state rankings, temporal dynamics, and multi-horizon validation (2019–2021 holdout) / future projections (2022+).

## Project Structure

```text
├── backend/
│   ├── analytics.py             # Core analytical methods, baseline growth, spatio-temporal aggregation
│   ├── analytics_ml.py          # ML model serving (LSTM, Autoencoder, Mahalanobis, PCA, Ensembles)
│   ├── app.py                   # Flask API application and web endpoints
│   └── ml_data_utils.py         # Outlier flagging and data repairing utilities
├── dataset/
│   └── cleaned_crime_data.csv   # Standardized NCRB panel dataset (36 States/UTs, 2001–2021)
├── frontend/
│   ├── css/style.css            # Responsive dashboard styles and themes
│   ├── index.html               # Main interactive web dashboard
│   └── js/app.js                # Frontend data visualization & Plotly/Leaflet rendering
├── models/
│   ├── cri_autoencoder.keras    # Trained Keras autoencoder weights
│   ├── cri_ensemble_meta.json   # PCA projection weights and evaluation metrics
│   ├── forecast_lstm.keras      # Trained global LSTM neural network
│   └── forecast_ensemble_meta.json # Normalization parameters & backtest error weights
├── script/
│   ├── train_cri_ensemble.py    # Training pipeline for CRI Autoencoder and PCA fusion
│   └── train_forecast_ensemble.py # Training pipeline for Global LSTM and ensemble evaluation
├── requirements.txt             # Python dependencies
└── run.py                       # Application entrypoint
```

## Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Sanjay-Chauhan52/Hybrid-Neural-Econometric-Forecasting-of-Crimes-Against-Women-in-India.git
cd Hybrid-Neural-Econometric-Forecasting-of-Crimes-Against-Women-in-India

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
python run.py
```
Open your browser at `http://127.0.0.1:5000` to access the interactive dashboard.

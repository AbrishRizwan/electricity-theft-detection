# ⚡ Smart Grid Electricity Theft Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)

A Streamlit app that flags suspicious electricity meters from daily consumption data using an XGBoost classifier on 13 engineered statistical features. Users can adjust the audit threshold, view a ranked inspection queue, download it as CSV, and inspect each meter's consumption curve.

🔗 **Live demo:** https://electricity-theft-detection-kw2zjrbxech93sjhz4nmn4.streamlit.app/
![App Preview](image app.png)

## 📌 Problem
Electricity theft and meter tampering cause revenue loss for utilities, and physical inspections are expensive. A ranked list of the most suspicious meters helps focus inspections where they matter most.

## 📂 Data
- Source: State Grid Corporation of China (SGCC) electricity theft dataset https://www.kaggle.com/datasets/bensalem14/sgcc-dataset
- 42,372 consumers, about 1,034 days of daily consumption each
- 3,615 consumers (8.5%) are labelled as theft, so the data is highly imbalanced
- The demo app uses 100 sample meters (`sample_smart_meter_data.csv`)

## 🛠️ Approach
1. **Cleaning:** linear interpolation of missing days, forward/back fill, remaining gaps set to 0
2. **Features (13):** mean, std, variance, median, min, max, coefficient of variation,
   peak-to-average ratio, skewness, kurtosis, zero-day count and ratio, 30-day drop ratio
3. **Model:** XGBoost classifier (180 trees, max depth 4, learning rate 0.05, L1/L2 regularization,
   row and column subsampling 0.8). Hyperparameters were set manually, no search was run.
4. **Imbalance:** handled with `scale_pos_weight` (negative/positive ratio) instead of resampling
5. **Threshold:** default 0.50; the app slider (0.30 to 0.80) lets users trade off precision and recall


## 📊 Results
Evaluated on a stratified random 80/20 split (test set: 8,475 meters, 723 theft cases).
**ROC-AUC: 0.77.** Accuracy is not a useful metric here, since predicting "normal" for every meter
would already score 91.5%.

| Threshold | Precision | Recall | Meters flagged |
|---|---|---|---|
| 0.3 | 0.122 | 0.896 | 5,302 |
| 0.4 | 0.150 | 0.787 | 3,806 |
| **0.5 (default)** | **0.195** | **0.672** | **2,488** |
| 0.6 | 0.246 | 0.534 | 1,569 |
| 0.7 | 0.305 | 0.382 | 906 |
| 0.8 | 0.410 | 0.196 | 346 |

**Key finding:** at the default threshold the model catches about 67% of theft cases, but only
about 1 in 5 flagged meters is actually theft (a random inspection would find 1 in 12). Raising the
threshold to 0.8 gives about 41% precision, roughly 5x better than random, at the cost of catching
only 20% of theft cases. The best threshold depends on the utility's inspection budget.

## ⚠️ Limitations & Next Steps
- Moderate performance (ROC-AUC 0.77): the model ranks suspicious meters better than chance but
  produces many false alarms, so flagged meters still need inspection
- Features are overall statistics per meter; monthly/weekly patterns and trend features are not used
- The 100-meter demo file comes from the same dataset the model was trained on, so demo predictions
  are not an independent test
- Next step: add time-based features and compare with LightGBM

## 📁 Repository Structure
```
├── app.py                                   # Streamlit app (inference and dashboard)
├── Electricity_Theft_Detection_Project.ipynb # Data cleaning, feature engineering, model training and evaluation
├── xgboost_optimized_detector.joblib        # Trained XGBoost model
├── sample_smart_meter_data.csv              # 100 randomly sampled meters for the demo
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

## ▶️ Run Locally
```bash
git clone https://github.com/AbrishRizwan/electricity-theft-detection.git
cd electricity-theft-detection
pip install -r requirements.txt
streamlit run app.py
```


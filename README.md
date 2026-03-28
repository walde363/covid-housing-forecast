# COVID-Era Structural Changes and Their Impact on Florida Housing Price Forecasting
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC6B23)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-2C5BB4)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-1A73E8)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

## Project Overview
The COVID-19 pandemic introduced significant changes to the U.S. housing market, including shifts in demand, migration patterns, and economic conditions. Florida experienced particularly strong housing price growth during this period, yet it remains unclear which factors most influenced this growth and whether these effects persist post-pandemic.

This project aims to identify the key drivers of housing price increases during the COVID-19 era and evaluate whether incorporating these factors improves forecasting accuracy for Florida’s housing market. Additionally, the study examines how national-level trends influence regional housing dynamics and predictive model performance.

---

## Objectives
- Identify key drivers of housing price growth during COVID-19  
- Evaluate the impact of structural changes on forecasting accuracy  
- Compare traditional, machine learning, and deep learning forecasting models  
- Analyze national vs. regional influences on Florida housing trends  

---

## Data Sources
This project integrates multiple public datasets capturing housing, economic, and environmental factors:

- **Housing Data:** Zillow, Redfin (market activity, inventory, pricing trends)  
- **Economic Indicators:** Freddie Mac mortgage rates, unemployment rates, building permits (FRED)  
- **Contextual Data:** ACS demographics, hurricane data (NOAA)  

**Key Features**
- Housing prices and inventory metrics  
- Mortgage rates and economic indicators  
- Regional and temporal identifiers (date, region/ZIP code)  

---

## Methodology
- Models will be trained using time-based train/test splits  
- Feature selection and handling of missing values will be applied to improve model robustness  

### Data Processing
- Merge multi-source datasets using time-based keys  
- Handle missing values through imputation and filtering  
- Feature engineering:
  - Lag features (1, 2, 3, 6, 12 months)  
  - Rolling statistics (means, standard deviations)  

### Models
A range of models will be used to capture different data characteristics:

| Category | Models |
|----------|--------|
| Baseline | Seasonal Naive |
| Time Series | SARIMAX, Prophet |
| Machine Learning | Random Forest, XGBoost |
| Deep Learning (Tentative) | LSTM, Temporal Fusion Transformer (TFT) |

- **SARIMAX / Prophet** → interpret trend and seasonality  
- **RF / XGBoost** → capture nonlinear relationships  
- **LSTM / TFT** → model long-term dependencies and multivariate interactions  

---

## Evaluation Metrics

| Metric | Purpose |
|--------|--------|
| RMSE | Penalizes large forecasting errors |
| MAE | Measures average prediction error |
| MAPE | Provides interpretable percentage error |
| MASE | Compares performance against baseline |

### Success Criteria
- Model outperforms **Seasonal Naive baseline** (MASE < 1)  
- MAPE ≤ 10–15% for forecasting accuracy  
- RMSE lower than baseline and comparable models  
- Stable performance across regions and time splits  

---

## Deliverables (Outputs)
- Cleaned and merged dataset  
- Trained forecasting models  
- Model comparison summary  
- Forecast visualizations  
- Interactive dashboard (Streamlit/web app)  
- Final report and presentation  

---

## Expected Outcomes
- Identification of key drivers of housing price growth during COVID-19  
- Determination of the most effective forecasting models  
- Insights into how structural changes influence regional housing markets  
- Evaluation of whether advanced ML/DL methods outperform traditional approaches  

---

## Implications
This project provides value to:
- Home buyers and real estate professionals  
- Investors and financial analysts  
- Policymakers and housing market researchers  

Understanding housing price drivers during COVID-era disruptions enables better forecasting, improved decision-making, and deeper insight into how structural shocks affect economic systems.

---

## Repository Structure
- /data → raw and processed datasets  
- /src → modeling scripts  
- /notebooks → exploration and experiments
- /reports → Models visualization and EDA

## Revisions Based on Peer Feedback
- Defined clear success criteria using MASE, RMSE, and MAPE  
- Improved explanation of evaluation metrics  
- Clarified handling of regions and geographic segmentation  
- Strengthened connection between models and project goals  

## Setup and Installation

### Clone the Repository
```bash
git clone https://github.com/walde363/covid-housing-forecast.git
cd covid-housing-forecast
```

### Create environment (RECOMMENDED)
```bash
conda create -n housing_env python=3.12
conda activate housing_env
```
or
```bash
python -m venv venv
# source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Dashboard (In progress)
```bash
streamlit run app/app.py
```

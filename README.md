# Financial Forecasting

## Overview
Predict tomorrow’s stock price movement (up or down) using past 30 days of data.  
Compare baseline models (Logistic Regression, Random Forest) with an advanced LSTM.

## Dataset
- Source: Kaggle financial market dataset  
- Columns: Date, Ticker, Open, High, Low, Close, Volume  
- Target: Binary (1 = tomorrow’s close > today’s, 0 = otherwise)

## Methods
- EDA and feature engineering (lag features, simple indicators)  
- Baseline: Logistic Regression, Random Forest  
- Advanced: LSTM sequence model  
- Metrics: Accuracy, Precision, Recall

## Results
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~0.65    |
| Random Forest       | ~0.66    |
| LSTM                | ~0.48    |

## Run
```bash
git clone https://github.com/yourusername/financial-forecasting.git
jupyter notebook 2. Notebooks/stock_prediction.ipynb

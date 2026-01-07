# Financial Forecasting: Baseline ML vs LSTM for Stock Movement Prediction

## Overview
This project predicts **next-day stock price direction (up/down)** and compares the performance of **classical machine learning models** against a **deep learning LSTM model**.  
The goal is to demonstrate proper modeling workflow, strong baselines, and critical evaluation in a financial time-series context.

---

## Problem Statement
Given historical stock market data, can we predict whether tomorrow’s closing price will be higher than today’s?

This is framed as a **binary classification problem**:
- `1` → Tomorrow’s close > Today’s close  
- `0` → Otherwise

---

## Dataset
- **Source:** Public financial market dataset (Kaggle)
- **Features:** Date, Ticker, Open, High, Low, Close, Volume
- **Target:** Directional price movement (binary)

---

## Project Structure
project/
├── data/
│   └── financial_forecasting_dataset.csv
├── notebooks/
│   └── stock_prediction.ipynb
├── results/
└── README.md



Paths in the notebook are **relative to the notebook location** to ensure the project runs immediately after cloning.

---

## Methodology
1. **Exploratory Data Analysis (EDA)**
   - Price trends and distribution analysis
   - Missing value handling

2. **Feature Engineering**
   - Lagged price features
   - Simple technical indicators

3. **Models**
   - **Baseline models**
     - Logistic Regression
     - Random Forest
   - **Sequence model**
     - LSTM (using historical closing prices)

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall  
   
   Emphasis is placed on comparing deep learning performance against strong baselines.

---

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~0.65    |
| Random Forest       | ~0.66    |
| LSTM                | ~0.48    |

**Key observation:**  
The LSTM did not outperform simpler models, highlighting the importance of strong baselines and the challenges of financial time-series prediction.

---

## Key Takeaways
- Deep learning does **not automatically outperform** classical models in financial data.
- Proper baselines are critical before introducing model complexity.
- Financial markets are noisy and difficult to predict using price data alone.

---

## Limitations & Future Work
- No transaction costs or trading strategy simulation
- Limited feature set (price-based only)
- Future improvements may include:
  - Additional technical indicators
  - Walk-forward (rolling) validation
  - Risk-adjusted performance metrics

---

## How to Run
```bash
git clone https://github.com/yourusername/financial-forecasting.git
cd financial-forecasting
jupyter notebook notebooks/stock_prediction.ipynb

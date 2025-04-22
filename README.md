# Forecasting Rice, Coffee, and Palm Oil Production in ASEAN Using Univariate LSTM

## 📘 **Project Overview**
This project focuses on forecasting the production of **rice**, **coffee**, and **palm oil** in four ASEAN countries—**Indonesia**, **Vietnam**, **Thailand**, and **Malaysia**—using **univariate LSTM models**. The goal is to provide accurate predictions for the years **2024–2030** based on historical production data (1961–2023) from the [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data).

---

## ❓ **Problem Statement**
1. Agricultural production in ASEAN countries is highly volatile, making it difficult for governments and stakeholders to plan long-term strategies.
2. There is a lack of standardized forecasts comparing production trends across ASEAN nations for key commodities.
3. Traditional forecasting methods fail to capture non-linear and seasonal patterns in agricultural time-series data.

---

## 🎯 **Project Goals**
1. Develop **univariate LSTM models** to forecast yearly production of rice, coffee, and palm oil for each country.
2. Compare projected production outcomes to evaluate Indonesia’s competitive position in ASEAN.
3. Improve forecast accuracy by modeling non-linear and seasonal trends using LSTM.

---

## 📊 **Dataset**
- **Source**: [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)
- **Time Period**: 1961–2023
- **Key Variables**:
  - `Rice Production`
  - `Coffee green Production`
  - `Palm oil Production`
- **Countries**: Indonesia, Vietnam, Thailand, Malaysia

---

## ⚙️ **Methodology**
1. **Data Preprocessing**:
   - Filtered data for selected countries and commodities.
   - Cleaned column names and handled outliers using log transformation.
   - Normalized data using MinMaxScaler.
   - Split data into training (80%) and testing (20%) sets chronologically.
   - Reshaped data into sequences for LSTM input.

2. **Modeling**:
   - Built separate **univariate LSTM models** for each country-commodity pair.
   - Configured models with:
     - `LSTM(units=50, activation='relu')`
     - `Dense(units=1)`
     - Loss: Mean Squared Error (MSE)
     - Optimizer: Adam
   - Applied EarlyStopping to prevent overfitting.

3. **Evaluation**:
   - Metrics: Root Mean Squared Error (RMSE) and Mean Squared Error (MSE).
   - Visualized predictions against actual values.

4. **Forecasting**:
   - Predicted production trends for 2024–2030.
   - Compared results across countries and commodities.

---

## 📈 **Results**
- **Best Performance**: Indonesia’s coffee production model (RMSE = 0.160).
- **High Volatility**: Vietnam’s palm oil and coffee models showed the highest RMSE (≈2.7).
- **Forecasts**:
  - Indonesia is projected to maintain dominance in palm oil production.
  - Vietnam and Thailand show potential growth in rice and coffee production.

---

## 📊 **Visualizations**
- **Outlier Handling**: Log-transformed data visualized with box plots.
- **Model Evaluation**: Forecasts plotted against actual production values.
- **Forecasting Results**: Bar charts and time-series plots for 2024–2030 projections.

---

## 🛠️ **Technologies Used**
- **Programming Language**: Python
- **Libraries**: TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 📂 **Project Structure**
- `Predictive_Analytics.ipynb`: Jupyter notebook for data preprocessing, modeling, and evaluation.
- `report_predictive_analytics.md`: Detailed project report.
- `README.md`: Project summary.

---
## 📉 **Result Prediction**

The project forecasts the production of rice, coffee, and palm oil for the years **2024–2030** across the four ASEAN countries. Below are the key predictions:
### Palm Oil Production
![Forecasted Palm Oil Production](repo_dir/Forecast-PalmOil-Until-2030.png)
### Coffee Production
![Forecasted Coffee Production](repo_dir/Forecast-CoffeGreen-Until-2030.png)
### Rice Production
![Forecasted Rice Production](repo_dir/Forecast-Rice-Until-2030.png)

These predictions provide valuable insights for stakeholders to plan agricultural strategies and benchmark production trends across ASEAN nations.

---
## 📌 **References**
1. [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)
2. [ML Models: Food Security and Climate Change](https://link.springer.com/chapter/10.1007/978-3-031-08743-1_6)
3. [Predicting Agricultural Commodities with Machine Learning](https://arxiv.org/abs/2310.18646)
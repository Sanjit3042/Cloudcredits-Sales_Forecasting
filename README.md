# Sales Forecasting

## Project Overview

This project aims to forecast future sales based on historical sales data using time series forecasting models. The models implemented include ARIMA, Exponential Smoothing (ETS), and machine learning-based forecasting. The project compares different techniques and evaluates their performance based on error metrics.

## Dataset

The dataset used for this project is a retail sales dataset, which includes information such as transaction date, customer demographics, product details, and total sales amount.

### Dataset Columns:
- **Transaction ID**: Unique identifier for each transaction.
- **Date**: Date of the transaction.
- **Customer ID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Product Category**: Category of the product sold.
- **Quantity**: Number of units sold.
- **Price per Unit**: Price of each unit sold.
- **Total Amount**: Total sales amount for the transaction (used as sales data for forecasting).

## Project Structure

```
Sales_forecasting/
│
├── data/
│   └── retail_sales_dataset.csv  # Dataset file (replace with the actual dataset name)
│
├── main.py  # Main Python script for sales forecasting
│
└── README.md  # This README file
```

## Requirements

The project requires the following Python packages:

- **pandas**: For data manipulation and preprocessing
- **matplotlib**: For plotting and visualizing the sales data and forecasts
- **statsmodels**: For ARIMA and Exponential Smoothing models
- **sklearn**: For machine learning models and evaluation metrics

You can install the required dependencies using:

```bash
pip install pandas matplotlib statsmodels scikit-learn
```

## How to Run

1. Clone or download this repository.
2. Download the retail sales dataset and place it in the `data/` folder.
3. Run the `main.py` script to perform sales forecasting using ARIMA, Exponential Smoothing, and Random Forest models.

```bash
python main.py
```

## Code Explanation

### 1. **Data Preprocessing**
   - The dataset is loaded and the `Date` column is converted to `datetime` format.
   - Missing values are handled using forward-filling (`ffill`).
   - The `Total Amount` column is used as the sales data for forecasting.

### 2. **ARIMA Model**
   - ARIMA is a popular time series forecasting model.
   - The order (p, d, q) is set as (5, 1, 0), and the model is trained on 80% of the data.
   - The model is evaluated using Root Mean Squared Error (RMSE).

### 3. **Exponential Smoothing (ETS)**
   - The ETS model uses additive trend and seasonal components to predict future sales.
   - It is also evaluated using RMSE.

### 4. **Random Forest (Machine Learning Model)**
   - A Random Forest model is trained using features extracted from the date (month and year).
   - This provides a machine learning-based approach to forecasting sales.

### 5. **Model Comparison**
   - The performance of each model is compared using RMSE.
   - Visualizations are generated to compare actual sales and predicted sales for each model.

## Output

The script will display:
- A time series plot of actual sales vs. predicted sales for each model.
- RMSE (Root Mean Squared Error) values for each model to evaluate accuracy.

## Example Plots

![Sales Forecast Plot](example_sales_forecast_plot.png)

## Improvements

1. **Model Optimization**: Experiment with different ARIMA and ETS parameters to improve accuracy.
2. **Additional Models**: Add other machine learning models such as XGBoost or deep learning models like LSTM for more advanced forecasting.
3. **Cross-Validation**: Implement cross-validation techniques to further evaluate the models' performance.

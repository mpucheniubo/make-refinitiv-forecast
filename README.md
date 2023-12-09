# 📊 Make Forecast

This is a small and quick implementation of several forecast methods for time series stock data obtained from the `Refinitiv` API that runs as on an Azure Function. The need to use Facebook's Prophet required a Dockerized solution.

The available methods are:

- Vector ARIMA
- FastAI
- Prophet
- LightGBM
- XGBoost
- NeuralProphet

## ⚙️ Functionality

With the payload of the HTTP request it is evaluated wether the user is active and had the option to run the forecasts. If so, an instance of the main object, `Framework`, is created. All pre-defined features are computed and the desired forecasting method is executed.

All the time series are detrended in the process and the values predicted are in fact the detrended ones. After the prediction has been made, the time series is rebuilt based on the predicted rates and the last available actual value.

Estimates for the uncertainty of the prediction are also provided.

## 📜 Notes

This was part of a larger project that never went into production, so a cleaner implementation with a more TDD approach won't happen.

The project is being made public without the git history.
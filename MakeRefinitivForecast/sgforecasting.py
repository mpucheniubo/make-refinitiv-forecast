import json
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import warnings

warnings.filterwarnings('ignore')

from statsmodels.tsa.vector_ar.var_model import VAR
from prophet import Prophet
import lightgbm as lgbm
import xgboost as xgb
from neuralprophet import NeuralProphet
from fastai.tabular.all import *

class Framework:

	def __init__(self,jsonObject:dict):
		self.Method = jsonObject["Method"]
		self.Scenario = jsonObject["Scenario"]
		self.HasCombined = jsonObject["HasCombined"]
		self.Dataset = self.__InitDataset(jsonObject)
		self.FeatureMethods = self.__InitFeatureMethods()
		self.ForecastMethods = self.__InitForecastMethods()
		
		self.HttpStatusCode = 200

	def __InitDataset(self,jsonObject:dict):
		return self._Dataset(self,jsonObject)

	def __InitFeatureMethods(self):
		return self._FeatureMethods(self)

	def __InitForecastMethods(self):
		return self._ForecastMethods(self)

	class _Dataset:

		def __init__(self,fw,jsonObject:dict):
			self.__Framework = fw
			
			self.TargetName = jsonObject["Target"]["MetaFields"]["NAME"]
			self.ValueColumns = ["Target"]
			self.FeaturesCategorical = []
			self.FeaturesNumerical = []

			self.Base = None
			self.Forecast = None
			self.Value = None

			self.ForecastBeginning = None
			self.ForecastEnding = None
			self.TimeHorizon = None

			self.GenerateBaseDataFrame(jsonObject)
			self.GenerateForecastDataFrame()
			self.GenerateTrainTestDataFrame()

		def GenerateBaseDataFrame(self,jsonObject:dict):
			_dtypes = {
						"TIMESTAMP" : "object",
						"CLOSE" : "float"
					}

			_target = jsonObject["Target"]

			df = pd.read_json(json.dumps(_target["Row"]), orient = "records", dtype = _dtypes)
			df.rename(columns = {"TIMESTAMP":"Date", "CLOSE":"Target"}, inplace = True)			
			df = Framework._Dataset.ParseToDate(df)

			if self.__Framework.HasCombined:

				_support = jsonObject["Support"]

				for it, supportObject in enumerate(_support):

					_supportName = f"Support_{it}"

					self.ValueColumns.append(_supportName)
					
					df_tmp = pd.read_json(json.dumps(supportObject["Row"]), orient = "records", dtype = _dtypes)
					df_tmp.rename(columns = {"TIMESTAMP":"Date", "CLOSE":_supportName}, inplace = True)
					df_tmp = Framework._Dataset.ParseToDate(df_tmp)
					
					df = df.merge(df_tmp, on = ["Date"], how = "left")

				df = df.dropna()
			
			df["IsTrain"] = 1

			self.Base = df
			self.ForecastBeginning = self.Base["Date"].iat[-1] + BDay(1)
			self.ForecastEnding = self.ForecastBeginning + BDay(50)

		def GenerateForecastDataFrame(self):
			self.Forecast = pd.DataFrame(pd.bdate_range(self.ForecastBeginning, self.ForecastEnding, freq = "B"), columns = ["Date"])
			self.Forecast["IsTrain"] = 0
			self.TimeHorizon = self.Forecast.shape[0]

		def GenerateTrainTestDataFrame(self):
			df_tmp = self.Base.append(self.Forecast).copy()

			df_tmp.sort_values(by = "Date", inplace = True)

			for column in self.ValueColumns:
				df_tmp["log|" + column] = df_tmp[column].apply(lambda x: np.log(x))
				df_tmp["lag|log|" + column] = df_tmp["log|" + column].shift(1)
				df_tmp["y|" + column] = df_tmp["log|" + column] - df_tmp["lag|log|" + column]

			# drop the first row, since it is empty for lag|log|* & y|*
			self.Value = df_tmp.tail(-1).reset_index(drop = True)

		def GetPredictionColumns(self):
			prediction_columns = []

			for column in self.Value.columns:
				if "y|" in column:
					prediction_columns.append(column)

			return prediction_columns

		def GetPrediction(self):
			return self.Value[lambda df: df["IsTrain"] == 0].copy()

		def GetTrain(self):
			return self.Value[lambda df: df["IsTrain"] == 1].copy()

		def GetValuesFromReturns(self):
			df_tmp = self.GetPrediction()

			y_0 = df_tmp["y|Target"].values
			y_1 = df_tmp["lag|log|Target"].values
			y_2 = np.zeros(self.TimeHorizon)

			sy = df_tmp["sy|Target"].values

			for it,y in enumerate(y_0):
				y_2[it] = y_1[it] + y
				if it < self.TimeHorizon - 1:
					y_1[it + 1] = y_2[it]

			self.Forecast["Target"] = np.exp(y_2)
			self.Forecast["SigmaTarget"] = 1.96*np.exp(sy)*np.sqrt(1 + np.arange(self.TimeHorizon))

		def GetResponseBody(self):
			df_tmp = self.Forecast.copy()

			# apply scenario growth
			df_tmp["Target"] *= (1 + 0.05*self.__Framework.Scenario)
			df_tmp["SigmaTarget"] *= (1 + 0.05*self.__Framework.Scenario)

			# restrict growth to 20%
			last_value = self.Base["Target"].iat[-1]
			max_abs_diff = (df_tmp["Target"]-last_value).abs().max()

			if max_abs_diff > 1.2*last_value:
				df_tmp["Target"] *= 1.2*last_value/max_abs_diff
				df_tmp["SigmaTarget"] *= 1.2*last_value/max_abs_diff


			df_tmp = self.Base.append(df_tmp)
			df_tmp = df_tmp[["Date","Target","SigmaTarget","IsTrain"]].fillna(0)
			df_tmp["Date"] = df_tmp["Date"].dt.strftime('%Y-%m-%d')
			return df_tmp.to_json(orient = "records")

		@staticmethod
		def ParseToDate(df:pd.DataFrame):
			df["Date"] = df["Date"].apply(lambda x: x[:10])
			df["Date"] = pd.to_datetime(df["Date"])
			df = df.sort_values(by = ["Date"])
			return df

	class _FeatureMethods:
		
		def __init__(self,fw):
			self.__Framework = fw

		def GenerateTimeFeatures(self):
			self.__Framework.Dataset.Value["f|cat|DayOfWeek"] = self.__Framework.Dataset.Value["Date"].dt.dayofweek
			self.__Framework.Dataset.Value["f|cat|DayOfYear"] = self.__Framework.Dataset.Value["Date"].dt.dayofyear
			self.__Framework.Dataset.Value["f|cat|Month"] = self.__Framework.Dataset.Value["Date"].dt.month
			self.__Framework.Dataset.Value["f|num|Year"] = self.__Framework.Dataset.Value["Date"].dt.year

			self.__Framework.Dataset.FeaturesCategorical.append("f|cat|DayOfWeek")
			self.__Framework.Dataset.FeaturesCategorical.append("f|cat|DayOfYear")
			self.__Framework.Dataset.FeaturesCategorical.append("f|cat|Month")
			self.__Framework.Dataset.FeaturesNumerical.append("f|num|Year")

		def GetFeatures(self):
			
			features = []

			for feature in self.__Framework.Dataset.FeaturesCategorical:
				features.append(feature)

			for feature in self.__Framework.Dataset.FeaturesNumerical:
				features.append(feature)

			return features

		@staticmethod
		def UnpivotPredictionColumns(df:pd.DataFrame, key_columns:list, value_columns:list):
			df_tmp = pd.melt(df, id_vars = key_columns, var_name = "d|cat|Target", value_name = "y|Target")
			df_values = df_tmp[["d|cat|Target"]].drop_duplicates().sort_values(by = ["d|cat|Target"]).copy()
			df_values["f|cat|Target"] = np.arange(df_values.shape[0])
			df_values["f|cat|Target"] = df_values["f|cat|Target"].astype(int)
			df_tmp = df_tmp.merge(df_values, on = ["d|cat|Target"])
			return df_tmp

		def RunAll(self):
			self.GenerateTimeFeatures()
	
	class _ForecastMethods:
		
		def __init__(self,fw):
			self.__Framework = fw

		def CallToARIMA(self):

			features = self.__Framework.FeatureMethods.GetFeatures()
			prediction_columns = self.__Framework.Dataset.GetPredictionColumns()

			df_t = self.__Framework.Dataset.GetTrain()
			df_p = self.__Framework.Dataset.GetPrediction()

			x_t, y_t = df_t[features], df_t[prediction_columns]
			x_p = df_p[features]

			print(y_t, x_t)

			model = VAR(
						y_t,
						exog = x_t
					)

			reg = model.fit(maxlags = 5, ic = 'aic')

			lag_order = reg.k_ar

			y_p = reg.forecast(
								y_t.values[-lag_order:], # starting dataset
								self.__Framework.Dataset.TimeHorizon, # steps
								exog_future = x_p # exogene variable
								)

			# first entry of the covarince matrix
			# take the sqrt to get the std
			sy_p = np.sqrt(reg.mse(self.__Framework.Dataset.TimeHorizon)[:,0,0])

			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, prediction_columns] = y_p
			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "sy|Target"] = sy_p 

		def CallToFastAI(self):
			
			seed_value = 42

			np.random.seed(seed_value) # cpu vars
			torch.manual_seed(seed_value) # cpu  vars
			random.seed(seed_value) # Python

			procs = [FillMissing, Categorify, Normalize]
   
			cat_vars = self.__Framework.Dataset.FeaturesCategorical

			cont_vars = self.__Framework.Dataset.FeaturesNumerical

			targets = self.__Framework.Dataset.GetPredictionColumns()

			df_t = self.__Framework.Dataset.GetTrain()
			df_p = self.__Framework.Dataset.GetPrediction()

			valid_idx = df_t[lambda df: df.index >= int(0.8*df.shape[0])].index.tolist()

			splits = IndexSplitter(valid_idx)(range_of(df_t))
			
			to = TabularPandas(df_t, y_names = targets, cat_names = cat_vars, cont_names = cont_vars, procs = procs, splits = splits)
			dls = to.dataloaders(bs=64)

			learn = tabular_learner(dls, [100,50,20], metrics=rmse)

			learn.fit(5,0.1)

			dl_p = learn.dls.test_dl(df_p)
			preds_p = learn.get_preds(dl = dl_p)
			y_p = to_np(preds_p[0])[:,0]

			dl_t = learn.dls.test_dl(df_t)
			preds_t = learn.get_preds(dl = dl_t)
			y_t = to_np(preds_t[0])[:,0]

			std = np.std(df_t["y|Target"].values-y_t)
			sy_p = std*np.ones(len(y_p))

			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "y|Target"] = y_p
			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "sy|Target"] = sy_p 

		def CallToProphet(self):

			x_t = self.__Framework.Dataset.GetTrain()
			x_t.rename(columns={"Date":"ds","y|Target":"y"}, inplace=True)
			x_t = x_t[["ds", "y"]]

			x_p = self.__Framework.Dataset.GetPrediction()
			x_p.rename(columns={"Date":"ds","y|Target":"y"}, inplace=True)
			x_p = x_p[["ds", "y"]]

			# values musst be nonnegative
			x_p["floor"] = 0

			model = Prophet(daily_seasonality = True)
			model.fit(x_t)
			
			y_p = model.predict(x_p)
			y_p = y_p[["ds","yhat","yhat_lower"]]
			y_p["syhat"] = y_p["yhat"]-y_p["yhat_lower"]

			std = y_p["syhat"].mean()
			sy_p = std*np.ones(len(y_p))

			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "y|Target"] = y_p["yhat"].values
			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "sy|Target"] = sy_p

		def CallToLightGBM(self):
			features = self.__Framework.FeatureMethods.GetFeatures()
			prediction_columns = self.__Framework.Dataset.GetPredictionColumns()

			df_t = self.__Framework.Dataset.GetTrain()
			df_p = self.__Framework.Dataset.GetPrediction()

			df_t = df_t[features + prediction_columns]
			df_p = df_p[features + prediction_columns]

			df_t = self.__Framework.FeatureMethods.UnpivotPredictionColumns(df_t, features, prediction_columns)
			df_p = self.__Framework.FeatureMethods.UnpivotPredictionColumns(df_p, features, prediction_columns)

			# override features and prediction columns after unpivoting
			prediction_columns = ["y|Target"]
			new_features = ["f|cat|Target"]

			for feature in features:
				new_features.append(feature)
			
			x_t, y_t = df_t[new_features], df_t[prediction_columns]
			x_p = df_p[new_features]

			d_train = lgbm.Dataset(x_t, label = y_t)

			params = {
				'boosting_type': 'gbdt',
				'objective': 'regression',
				'metric': 'l2',
				'learning_rate': 0.5,
				'min_child_weight':1.5,
				'subsample':0.6,
				'max_depth':3,
				'colsample_bytree':0.4,
				'reg_alpha':0.75,
				'reg_lambda':0.45,
				'n_estimators':50,
				'verbose': -1
				}

			reg = lgbm.train(
							params,
							d_train,
							verbose_eval = False)

			y_p = reg.predict(x_p)

			df_p["y|Target"] = y_p

			y_p = df_p[lambda df: df["d|cat|Target"] == "y|Target"]["y|Target"].values

			df_t["y|Pred"] = reg.predict(x_t)

			df_e = df_t[lambda df: df["d|cat|Target"] == "y|Target"][["y|Target","y|Pred"]].copy()

			std = np.std(df_e["y|Target"]-df_e["y|Pred"])

			sy_p = std*np.ones(len(y_p))

			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "y|Target"] = y_p
			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "sy|Target"] = sy_p
		
		def CallToNeuralProphet(self):
			x_t = self.__Framework.Dataset.GetTrain()
			x_t.rename(columns={"Date":"ds","y|Target":"y"}, inplace=True)
			x_t = x_t[["ds", "y"]]

			model = NeuralProphet(daily_seasonality = True)
			model.fit(x_t, epochs=20, freq="B")

			y_p = model.make_future_dataframe(x_t, periods = self.__Framework.Dataset.TimeHorizon)
			y_p = model.predict(y_p)
			
			y_t = model.predict(x_t)
			x_t = x_t.merge(y_t[["ds","yhat1"]], on = ["ds"], how = "left")
			x_t["yhat1"] = x_t["yhat1"].interpolate()

			std = np.std(x_t["y"]-x_t["yhat1"])
			sy_p = std*np.ones(len(y_p))

			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "y|Target"] = y_p["yhat1"].values
			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "sy|Target"] = sy_p

		def CallToXGBoost(self):
			features = self.__Framework.FeatureMethods.GetFeatures()
			prediction_columns = self.__Framework.Dataset.GetPredictionColumns()

			df_t = self.__Framework.Dataset.GetTrain()
			df_p = self.__Framework.Dataset.GetPrediction()

			df_t = df_t[features + prediction_columns]
			df_p = df_p[features + prediction_columns]

			df_t = self.__Framework.FeatureMethods.UnpivotPredictionColumns(df_t, features, prediction_columns)
			df_p = self.__Framework.FeatureMethods.UnpivotPredictionColumns(df_p, features, prediction_columns)

			# override features and prediction columns after unpivoting
			prediction_columns = ["y|Target"]
			new_features = ["f|cat|Target"]

			for feature in features:
				new_features.append(feature)
			
			x_t, y_t = df_t[new_features], df_t[prediction_columns]
			x_p = df_p[new_features]

			d_t = xgb.DMatrix(x_t, label = y_t)
			d_p = xgb.DMatrix(x_p)

			params = {
				# 'max_depth': 2,
				'eta': 0.5,
				'objective': 'reg:squarederror',
				# 'colsample_bytree':0.4,
				# 'num_boost_round':50,
				# 'alpha':0.75,
				# 'lambda':0.45,
				# 'subsample':0.6,
				# 'min_child_weight':1.5,
                # 'n_estimators':1000
				}

			metrics = {}

			reg = xgb.train(
					params = params,
					dtrain = d_t,
					verbose_eval= False
					)

			y_p = reg.predict(d_p)

			df_p["y|Target"] = y_p

			y_p = df_p[lambda df: df["d|cat|Target"] == "y|Target"]["y|Target"].values

			df_t["y|Pred"] = reg.predict(d_t)

			df_e = df_t[lambda df: df["d|cat|Target"] == "y|Target"][["y|Target","y|Pred"]].copy()

			std = np.std(df_e["y|Target"]-df_e["y|Pred"])

			sy_p = std*np.ones(len(y_p))

			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "y|Target"] = y_p
			self.__Framework.Dataset.Value.loc[self.__Framework.Dataset.Value["IsTrain"] == 0, "sy|Target"] = sy_p 

		def CallMethod(self):
			_method = "self.CallTo" + self.__Framework.Method + "()"
			eval(_method)

	
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import scipy
import statsmodels.api as sm


class ZipCode_TSA:

    def __init__(self, df, zip_code, start_date, dynamic=False):
        self.df = df
        self.zip_code = zip_code
        self.start_date = start_date
        self.dynamic = dynamic

    def df_zipcode(self):
        """Returns DF of the requested zip code"""
        zipcode_df = self.df[self.df['ZipCode'] == self.zip_code]
        zipcode_df.set_index('time', inplace=True)
        zipcode_df = zipcode_df[self.start_date:]
        return zipcode_df

    # def pdq(self):
    #     p = d = q = range(2,3)
    #     pdq = list(itertools.product(p, d, q))
    #     pdqs = [(x[0], x[1], x[2], 1) for x in itertools.product(p, d, q)]
    #     ans = []
    #     df = self.df_zipcode()['value']
    #     for comb in pdq:
    #         for combs in pdqs:
    #             mod = sm.tsa.statespace.SARIMAX(df,
    #                                             order=comb,
    #                                             seasonal_order=combs,
    #                                             enforce_stationarity=False,
    #                                             enforce_invertibility=False)
    #             output = mod.fit()
    #             ans.append({"comb" : comb, "combs" : combs, "aic" : output.aic})
    #     return ans

    def arima_model(self):
        df = self.df_zipcode()['value']
        ARIMA_MODEL = sm.tsa.statespace.SARIMAX(df,
                                order=[2, 2, 2],
                                seasonal_order=[2, 2, 2, 1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        output = ARIMA_MODEL.fit()
        return output

    def self_prediction(self):
        pred = self.arima_model().get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
        pred_self_conf = pred.conf_int()
        return pred_self_conf

    def prediction_conf(self, steps=6):
        # Get forecast 6 steps ahead in future
        prediction = self.arima_model().get_forecast(steps)
        # Get confidence intervals of forecasts
        pred_conf = prediction.conf_int()
        return (prediction, pred_conf)

    def forecast(self):
        # mean_forecast = self.prediction.predicted_mean
        # What the forecast mean is at the end of 6 months
        conf_interval = self.prediction_conf()
        target = conf_interval.predicted_mean[-1] - conf_interval.predicted_mean[0]
        # max_gain = pred_conf.iloc[:, 1] - rk_1_2011['value'][-1]
        # max_loss = pred_conf.iloc[:, 0] - rk_1_2011['value'][-1]
        return target
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import scipy
import statsmodels.api as sm

class ZipCode_TSA():

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
    #     p = d = q = range(2, 3)
    #     pdq = list(itertools.product(p, d, q))
    #     pdqs = [(x[0], x[1], x[2], 1) for x in list(itertools.product(p, d, q))]
    #     ans = []
    #     for comb in pdq:
    #         for combs in pdqs:
    #             try:
    #                 mod = sm.tsa.statespace.SARIMAX(self.df_zipcode()['value'],
    #                                                 order=comb,
    #                                                 seasonal_order=combs,
    #                                                 enforce_stationarity=False,
    #                                                 enforce_invertibility=False)
    #                 output = mod.fit()
    #                 ans.append([comb, combs, output.aic])
    #                 # print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, combs, output.aic))
    #             except:
    #                 continue
    #     ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    #     pdq_pdqs = ans_df.loc[ans_df['aic'].idxmin()]   
    #     return (pdq_pdqs['pdq'], pdq_pdqs['pdqs'])

    def arima_model(self):
        # params = self.pdq()
        ARIMA_MODEL = sm.tsa.statespace.SARIMAX(self.df_zipcode()['value'],
                                order=[2, 2, 2],
                                seasonal_order=[2, 2, 2, 1],
                                enforce_stationarity=False,
                                enforce_invertibility=False, 
                                freq='MS')
        output = ARIMA_MODEL.fit()
        return output

    # def self_prediction(self):
    #     pred = self.arima_model().get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
    #     pred_self_conf = pred.conf_int()
    #     return pred_self_conf

    def prediction_conf(self, steps=16):
        # Get forecast 6 steps ahead in future
        prediction = self.arima_model().get_forecast(steps)
        # Get confidence intervals of forecasts
        pred_conf = prediction.conf_int()
        return (prediction, pred_conf)

    def forecast(self):
        # mean_forecast = self.prediction.predicted_mean
        # What the forecast mean is at the end of 6 months
        pred = self.prediction_conf()
        zipcode_df = self.df_zipcode()
        target = pred[0].predicted_mean[-1] - pred[0].predicted_mean[0]
        max_gain = pred[1].iloc[:, 1]['2018-10-01'] - zipcode_df['value'][-1]
        max_loss = pred[1].iloc[:, 0]['2018-10-01'] - zipcode_df['value'][-1]
        start_point = zipcode_df['value'][-1]
        result = (target, max_gain, max_loss, start_point)
        return result
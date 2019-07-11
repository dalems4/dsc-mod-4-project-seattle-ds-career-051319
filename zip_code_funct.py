import pandas as pd
import matplotlib.pyplot as plt
import itertools 
%matplotlib inline

class ZipCode_TSA(object):

    def __init__(self, zip_code, start_date, dynamic=False):
        self.zip_code = zip_code
        self.start_date = start_date
        self.dynamic = dynamic

    def df_zipcode(self):
        """Returns DF of the requeested zip code"""
        zipcode_df = self.zip_code[self.zip_code['ZipCode'] == self.zip_code]
        zipcode_df.set_index('time', inplace=True)
        zipcode_df = zipcode_df[start_date:]
        return zipcode_df

    def pdq(self):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        pdqs = [(x[0], x[1], x[2], 1) for x in list(itertools.product(p, d, q))]
        ans = []
        for comb in pdq:
            for combs in pdqs:
                try:
                    mod = sm.tsa.statespace.SARIMAX(rk_1_2011['value'],
                                                    order=comb,
                                                    seasonal_order=combs,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    output = mod.fit()
                    ans.append([comb, combs, output.aic])
                    print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, combs, output.aic))
                except:
                    continue
        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
        ans_df.loc[ans_df['aic'].idxmin()]   

    def arima_model(self):
        ARIMA_MODEL = sm.tsa.statespace.SARIMAX(rk_1_2011['value'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 1),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

        output = ARIMA_MODEL.fit()

    def prediction():
        pred = output.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
        pred_conf = pred.conf_int()

    
    def prediction_conf(self, steps=6):
        # Get forecast 6 steps ahead in future
        prediction = output.get_forecast(steps)
        # Get confidence intervals of forecasts
        pred_conf = prediction.conf_int()

    def forecast(self):
        mean_forecast = prediction.predicted_mean
        #What the forecast mean is at the end of 6 months
        prediction.predicted_mean[-1] - prediction.predicted_mean[0]
        max_gain = pred_conf.iloc[:, 1] - rk_1_2011['value'][-1]
        max_loss = pred_conf.iloc[:, 0] - rk_1_2011['value'][-1]
import numpy as np
import pandas as pd
import darts
from darts import TimeSeries
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, r2_score, rmse
from darts.models import TFTModel
from tqdm import tqdm
import numpy as np
from darts.utils.likelihood_models import QuantileRegression
import sys

quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]

def load_data():
    df = pd.read_csv('../input/rossmann-store-sales/train.csv', parse_dates=['Date'])
    df = pd.concat([df.drop(columns='StateHoliday'), pd.get_dummies(df.StateHoliday, prefix='Holiday')], axis=1)
    return df

def get_sample(df, num):
    sample = df[df.Store == num]
    series = TimeSeries.from_dataframe(sample, 'Date', 'Sales')

    train, test = series.split_before(pd.Timestamp("20150601"))

    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)

    series_customers = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='Customers')
    series_open = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='Open')
    series_promo = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='Promo')
    series_school = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='SchoolHoliday')
    series_weekday = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='DayOfWeek')
    series_holiday_a = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='Holiday_a')
    series_holiday_b = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='Holiday_b')
    series_holiday_c = TimeSeries.from_dataframe(sample, time_col='Date', value_cols='Holiday_c')

    customers_transformed = transformer.transform(series_customers)

    covariates = series_customers.stack(series_open)
    covariates = covariates.stack(series_promo)
    covariates = covariates.stack(series_school)
    covariates = covariates.stack(series_weekday)
    covariates = covariates.stack(series_holiday_a)
    covariates = covariates.stack(series_holiday_b)
    covariates = covariates.stack(series_holiday_c)

    train_covariates, test_covariates = covariates.split_before(pd.Timestamp("20150601"))
    return train_transformed, test_transformed, covariates

def rmspe(actual, pred):
    return np.sqrt(np.mean( ((actual - pred) / actual)**2)) 

def smape(actual, pred):
    return np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred))/2)) * 100

def evaluate_covariates_model(model, train, test, covariates=None):
    if covariates is not None:
        model.fit(
            train,
            future_covariates=covariates,
            verbose=False,
        )
    else:
        model.fit(
            train,
            verbose=False,
        )
    if covariates is not None:
        pred = model.predict(len(test), future_covariates=covariates)
    else:
        pred = model.predict(len(test))
    pred = pred.pd_dataframe()
    test = test.pd_dataframe()

    test.columns = ['Actual']
    pred.columns = ['Pred']

    df = pd.concat([test, pred], axis=1)
    df = df[df.Actual > 0]
    return smape(df.Actual.values, df.Pred.values), rmspe(df.Actual.values, df.Pred.values)

if __name__ == "__main__":

    store_num_list = pd.read_csv('../input/store_list.csv').Store.values

    print(store_num_list)

    df = load_data()

    smape_list = []
    rmspe_list = []

    for num in tqdm(store_num_list):
        train_transformed, test_transformed, covariates = get_sample(df, num)
        tft_model = TFTModel(
            input_chunk_length=28,
            output_chunk_length=7,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=16,
            n_epochs=100,
            add_relative_index=False,
            add_encoders=None,
            likelihood=QuantileRegression(
                    quantiles=quantiles
            ),  
            random_state=42, 
            pl_trainer_kwargs ={"accelerator": "gpu", "devices": [0], "enable_progress_bar": False}
        )
        
        smape_score, rmspe_score = evaluate_covariates_model(tft_model, train_transformed, test_transformed, covariates )
        rmspe_list.append(rmspe_score)
        smape_list.append(smape_score)
        print(num, smape_score, rmspe_score)
    
    pd.DataFrame(
    {
        'Store' : store_num_list,
        'rmspe' : rmspe_list,
        'smape' : smape_list,
    }).to_csv('tft_result_test.csv')

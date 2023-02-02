import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import darts
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, r2_score, rmse
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.likelihood_models import GaussianLikelihood
from darts.models import (
    AutoARIMA,
    Prophet,
    RNNModel,
    NBEATSModel,
    TFTModel,
)
import warnings
warnings.filterwarnings('ignore')


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

def evaluate_model(model, train, test, covariates=None):
    if covariates is not None:
        if model.__class__.__name__ == 'NBEATSModel':
            model.fit(
                train,
                past_covariates=covariates,
            )
        else:
            model.fit(
                train,
                future_covariates=covariates,
            )
    else:
        model.fit(train)
    
    if covariates is not None:
        if model.__class__.__name__ == 'NBEATSModel':
            pred = model.predict(len(test), past_covariates=covariates)
        else:
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

def get_model(model_name):
    if model_name == 'ARIMA':
        model = AutoARIMA()

    elif model_name == 'PROPHET':
        model = Prophet()

    elif model_name == 'LSTM':
        model = model = RNNModel(
            model="LSTM",
            input_chunk_length=28,
            random_state=42,
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0], "enable_progress_bar": False}
        )

    elif model_name == 'DEEPAR':
        model = RNNModel(
            model="LSTM",
            input_chunk_length=28,
            random_state=42,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0],"enable_progress_bar": False}
        )

    elif model_name == 'NBEATS':
        model = NBEATSModel(
            input_chunk_length=28,
            output_chunk_length=7,
            random_state=42,
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0],"enable_progress_bar": False}
        )

    elif model_name == 'TFT':
        model = TFTModel(
            input_chunk_length=28,
            output_chunk_length=7,
            random_state=42, 
        pl_trainer_kwargs ={"accelerator": "gpu", "devices": [0], "enable_progress_bar": False}
        )

    else:
        print('Model Name must be one of [ARIMA, PROPHET, NBEATS, DEEPAR, LSTM, TFT]')
        sys.exit()

    print(model.__class__.__name__)
    return model 


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print('''Usage : python model_train.py [Model_Name] [is Covariates] \n 
                ex) python model_train.py ARIMA Y \n 
                model_name : ARIMA, PROPHET, NBEATS, DEEPAR, LSTM, TFT''' )
        sys.exit()

    model_name = sys.argv[1]
    with_covariates = sys.argv[2]
    if with_covariates == 'Y':
        postfix = 'with_covs'
    else:
        postfix = 'without_covs'

    smape_list = []
    rmspe_list = []

    store_num_list = pd.read_csv('../input/store_list.csv').Store.values
    df = load_data()
    model = get_model(model_name)

    for num in tqdm(store_num_list[:2]):
        train_transformed, test_transformed, covariates = get_sample(df, num)  
        if with_covariates == 'N': 
            smape_score, rmspe_score = evaluate_model(model, train_transformed, test_transformed)
        elif with_covariates == 'Y':
             smape_score, rmspe_score = evaluate_model(model, train_transformed, test_transformed, covariates)
        rmspe_list.append(rmspe_score)
        smape_list.append(smape_score)
        print(num, smape_score, rmspe_score)

    result = pd.DataFrame(
        {
            'store' : store_num_list[:2],
            'rmspe' : rmspe_list,
            'smape' : smape_list,
        }
    )
    result.to_csv(f'../output/{model_name}_{postfix}_result.csv')

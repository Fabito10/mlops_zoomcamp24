```python
! pip install pyarrow
```

    Requirement already satisfied: pyarrow in /home/ubuntu/anaconda3/lib/python3.11/site-packages (14.0.2)
    Requirement already satisfied: numpy>=1.16.6 in /home/ubuntu/anaconda3/lib/python3.11/site-packages (from pyarrow) (1.26.4)



```python
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
```


```python
def download_data(url, filename):
    df = pd.read_parquet(url)
    df.to_parquet(filename)
    return df

def load_data(filename):
    return pd.read_parquet(filename)

def compute_duration(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    return df

def filter_outliers(df, min_duration=1, max_duration=60):
    return df[(df['duration'] >= min_duration) & (df['duration'] <= max_duration)]

def one_hot_encode(df, columns):
    df.loc[:, columns] = df[columns].astype(str)  # Use .loc to avoid SettingWithCopyWarning
    data_dicts = df[columns].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(data_dicts)
    return X, dv

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def calculate_rmse(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse

def main():
    # URLs for the datasets
    url_january = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    url_february = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet"

    # Filenames for local storage
    filename_january = "./data/yellow_tripdata_2023-01.parquet"
    filename_february = "./data/yellow_tripdata_2023-02.parquet"

    # Download and load the data
    df_january = download_data(url_january, filename_january)
    df_february = download_data(url_february, filename_february)

    # Question 1
    num_columns_january = df_january.shape[1]
    print(f"Number of columns in January data: {num_columns_january}")

    # Question 2
    df_january = compute_duration(df_january)
    std_duration = df_january['duration'].std()
    print(f"Standard deviation of trip duration in January: {std_duration:.2f}")

    # Question 3
    df_january_filtered = filter_outliers(df_january)
    fraction_remaining = len(df_january_filtered) / len(df_january)
    print(f"Fraction of records remaining after filtering outliers: {fraction_remaining:.2%}")

    # Question 4
    columns_to_encode = ['PULocationID', 'DOLocationID']
    X_january, dv = one_hot_encode(df_january_filtered, columns_to_encode)
    dimensionality = X_january.shape[1]
    print(f"Dimensionality of the feature matrix: {dimensionality}")

    # Question 5
    y_january = df_january_filtered['duration'].values
    model = train_model(X_january, y_january)
    rmse_train = calculate_rmse(model, X_january, y_january)
    print(f"RMSE on training data: {rmse_train:.2f}")

    # Question 6
    df_february = compute_duration(df_february)
    df_february_filtered = filter_outliers(df_february)
    X_february = dv.transform(df_february_filtered[columns_to_encode].to_dict(orient='records'))
    y_february = df_february_filtered['duration'].values
    rmse_val = calculate_rmse(model, X_february, y_february)
    print(f"RMSE on validation data: {rmse_val:.2f}")

if __name__ == "__main__":
    main()

```

    Number of columns in January data: 19
    Standard deviation of trip duration in January: 42.59
    Fraction of records remaining after filtering outliers: 98.12%
    Dimensionality of the feature matrix: 515
    RMSE on training data: 7.65
    RMSE on validation data: 13.32



```python

```


```python

```

# NYC Taxi Duration Prediction

This project involves analyzing and predicting the duration of yellow taxi trips in New York City using data from January and February 2021. The analysis includes steps such as downloading data, computing trip durations, filtering outliers, one-hot encoding, training a linear regression model, and evaluating the model's performance.

## Project Structure

- `homework_duration_prediction_yellow_taxi.ipynb`: Jupyter notebook containing the code for the project.
- `yellow_tripdata_2021-01.parquet`: Dataset for January 2021.
- `yellow_tripdata_2021-02.parquet`: Dataset for February 2021.

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- pyarrow

You can install the necessary libraries using `pip`:

```bash
pip install pandas numpy scikit-learn pyarrow


Running the Script
To run the script, use the following command:
python homework_duration_prediction_yellow_taxi.py

Functions
The script includes the following functions:

download_data(url, filename): Downloads and saves data from a given URL.
load_data(filename): Loads data from a given filename.
compute_duration(df): Computes the duration of trips in minutes.
filter_outliers(df, min_duration=1, max_duration=60): Filters out trips with durations outside the specified range.
one_hot_encode(df, columns): One-hot encodes specified columns in the dataframe.
train_model(X, y): Trains a linear regression model on the given data.
calculate_rmse(model, X, y): Calculates the Root Mean Squared Error (RMSE) for the given model and data.

Analysis Steps
Downloading and Loading Data: The script downloads and loads data for January and February 2021.
Computing Duration: The script computes the duration of each trip in minutes.
Filtering Outliers: Trips with durations less than 1 minute or more than 60 minutes are filtered out.
One-Hot Encoding: Pickup and dropoff location IDs are one-hot encoded.
Training a Model: A linear regression model is trained on the January data.
Evaluating the Model: The model's performance is evaluated on the February data.

The script prints the following information:
Number of columns in the January data.
Average duration of trips in January.
Fraction of records remaining after filtering outliers.
Dimensionality of the feature matrix.
RMSE on the training data.
RMSE on the validation data.


Example Output
Number of columns in January data: 19
Average duration of trip in January: 16.33
Fraction of records remaining after filtering outliers: 95.45%
Dimensionality of the feature matrix: 515
RMSE on training data: 7.64
RMSE on validation data: 7.81

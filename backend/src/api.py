from botocore.exceptions import ClientError

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .models import ProcessedDataResponse
from src.ml_pipeline.helper_functions import preprocess_data, scale_data, plot_daily_predictions

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as py

import boto3
import datetime
import io
import pickle


s3 = boto3.client("s3", region_name="us-east-1")
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
bucket_name = "cgm-analytics-ucb"

# Configure CORS
origins = [
    "https://www.nutrinet-ai.com",
    "http://www.nutrinet-ai.com",
    "http://api.nutrinet-ai.com",
    "https://api.nutrinet-ai.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def preprocess_feature_set(preprocess_data: preprocess_data, filename: str):

    """Calls preprocessing function that modifies the adults raw CGM 
       data into a standardized, reduced feature set for prediction."""

    BUCKET_NAME = "nutrinet-ml-models"
    SCALAR = pickle.loads(s3.get_object(Bucket=BUCKET_NAME, Key="standard_scaler_mealDetection.pickle")['Body'].read())

    download_file_from_s3(bucket_name, filename)
    df = pd.read_csv(filename)

    reduced_feature_set = preprocess_data(df)
    scaled_feature_set = scale_data(reduced_feature_set, SCALAR)

    return scaled_feature_set


def meal_prediction(filename: str):

    BUCKET_NAME = "nutrinet-ml-models"
    RFE_RESULTS = s3.get_object(Bucket=BUCKET_NAME, Key="lgbm_features_20230716.csv")['Body'].read()
    LGBM_MEAL_DETECTION_MODEL = pickle.loads(s3.get_object(Bucket=BUCKET_NAME, Key="lgbm_mealDetection_model.pickle")['Body'].read())
    scaled_feature_set = preprocess_feature_set(preprocess_data=preprocess_data, filename=filename)

    str_results = RFE_RESULTS.decode('utf-8')
    data = io.StringIO(str_results)
    df = pd.read_csv(data)

    selected_features = df.feature.to_list()
    X = scaled_feature_set.iloc[:, :-5]
    X = X[selected_features]

    preds = LGBM_MEAL_DETECTION_MODEL.predict(X)
    scaled_feature_set['predictions'] = preds

    return scaled_feature_set


def carb_prediction(filename: str):

    BUCKET_NAME = "nutrinet-ml-models"
    CARB_ESTIMATE_FEATURES = s3.get_object(Bucket=BUCKET_NAME, Key="svr_features_20230723.csv")['Body'].read()
    SVR_CARB_MODEL = pickle.loads(s3.get_object(Bucket=BUCKET_NAME, Key="svr_model_carbEstimate.pickle")['Body'].read())

    carb_feature_set = meal_prediction(filename)
    meal_data = carb_feature_set[carb_feature_set.predictions == 1]

    carb_df = pd.read_csv(io.BytesIO(CARB_ESTIMATE_FEATURES))
    carbEst_features = carb_df.feature.to_list()

    X = meal_data[carbEst_features]
    preds = SVR_CARB_MODEL.predict(X)
    meal_data['carb_preds'] = preds

    return meal_data


def generate_plot(df, filename=None):

    df['Day'] = pd.to_datetime(df['Day']).dt.date

    if df['Time'].dtype != 'datetime64[ns]':
    # Convert 'Time' column to datetime format and extract time
        df['Time'] = pd.to_datetime(df['Time']).dt.time

    # Combine 'Day' and 'Time' into a datetime column
    df['Datetime'] = df.apply(lambda row: datetime.datetime.combine(row['Day'], row['Time']), axis=1)
    df['Time'] = df['Datetime'].dt.round('5min').dt.time
    df['Minutes'] = df['Time'].apply(lambda x: x.hour * 60 + x.minute)

    # Group the data by 'Minutes' and calculate the percentiles
    data_grouped = df.groupby('Minutes')['CGM'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()

    # Create the base line (median)
    line = go.Scatter(
        x=data_grouped.index,
        y=data_grouped[0.50],
        mode='lines',
        line=dict(color='black'),
        name='Median'
    )

    # Create the first fill area (10% - 90%)
    fill_10_90 = go.Scatter(
        x=data_grouped.index.tolist() + data_grouped.index.tolist()[::-1],  # x, then x reversed
        y=data_grouped[0.10].tolist() + data_grouped[0.90].tolist()[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(211,211,211,0.5)',  # lightgray
        line=dict(color='rgba(255,255,255,0)'),  # lines between points are white
        showlegend=False,
        name='10-90 percentile'
    )

    # Create the second fill area (25% - 75%)
    fill_25_75 = go.Scatter(
        x=data_grouped.index.tolist() + data_grouped.index.tolist()[::-1],
        y=data_grouped[0.25].tolist() + data_grouped[0.75].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(105,105,105,0.5)',  # darkgray
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='25-75 percentile'
    )

    # Define layout
    layout = go.Layout(
        title='Ambulatory Glucose Profile',
        autosize=True,
        width=1500,
        height=500,
        xaxis=dict(
            title='Time',
            tickvals=np.arange(0, 24*60, 120),
            ticktext=[f'{h:02d}:00' for h in range(0, 24, 2)],
            tickangle=45
        ),
        yaxis=dict(
            title='Glucose Level'
        )
    )

    fig = go.Figure(data=[fill_10_90, fill_25_75, line], layout=layout)

    # Convert the figure to an HTML string
    fig_html = py.to_html(fig, full_html=True)

    # Save the HTML string as an HTML file
    with open(filename, 'w') as f:
        f.write(fig_html)


def create_bucket(bucket_name):
    try:
        response = s3.create_bucket(
            Bucket=bucket_name,
        )
        print(f"Bucket {bucket_name} created successfully.")
        return response
    except ClientError as e:
        print(e)
        return None


def upload_file_to_s3(bucket_name, file_name, file_data):
    try:
        # Check if file_data is a path (str) or a file-like object
        if isinstance(file_data, str):
            # It's a path, so open the file
            with open(file_data, "rb") as f:
                ### Todo - Need to write the logic to handle static images or plotly html which uses ExtraArgs={'ContentType': 'text/html'}
                s3.upload_fileobj(f, bucket_name, file_name, ExtraArgs={'ContentType': 'image/png'})
        else:
            # It's a file-like object, so ensure we're at the start before uploading
            file_data.seek(0)
            s3.upload_fileobj(file_data, bucket_name, file_name)
        print(f"File {file_name} uploaded successfully to bucket {bucket_name}")
        return True
    except ClientError as e:
        print(
            f"An error occurred while uploading {file_name} to {bucket_name}. Error: {e}"
        )
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.post("/api/analysis", response_model=ProcessedDataResponse)
async def process_csv_file(file: UploadFile):
    try:
        s3.head_object(Bucket=bucket_name, Key=file.filename)
        return ProcessedDataResponse(message="Success: File already present.", file=file.filename)

    except ClientError as e:
        # If the file doesn't exist, upload it
        if e.response["Error"]["Code"] == "404":
            print("Data not in S3 bucket - Uploading...")
            file.file.seek(0)
            upload_file_to_s3(bucket_name, file.filename, file.file)
            return ProcessedDataResponse(message="Success: File uploaded.", file=file.filename)
        else:
            raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def check_if_file_exists(bucket: str, key: str):
    """Helper function used with GET request to validate the file of interest is in the S3 bucket."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            print(f"Unexpected error: {e}")
        return False


def download_file_from_s3(bucket: str, key: str):
    """Helper function - once we confirm the file we want is located in the S3 bucket, pull it down for transformation"""
    try:
        s3.download_file(bucket, key, key)
        print(f"File downloaded successfully from S3")
    except ClientError as e:
        print(f"Couldn't download the file: {e}")


@app.get("/api/analysis/{file_name}")
async def get_csv_file(file_name: str):
    try:
        if not check_if_file_exists(bucket_name, file_name):
            raise HTTPException(status_code=404, detail="File not found")

        download_file_from_s3(bucket_name, file_name)

        df = pd.read_csv(file_name)
        records = df.head(200).to_dict(orient="records")

        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualization/{filename}")
async def get_visualization(filename: str):

    # Check if the image exists in S3
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{filename}_prediction.png")
        prediction_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": f"{filename}_prediction.png"},
            ExpiresIn=3600,
        )
        print("Located previously generated prediction. Loading into frontend...")
    except ClientError:

        print("Loading carb data...")
        predicted_meal_data = carb_prediction(filename=filename)

        print("Loading prediction plot...")
        plot_daily_predictions(predicted_meal_data, "carbPrediction.png")

        print("Uploading Prediction to S3 bucket...")
        upload_file_to_s3(bucket_name, f"{filename}_prediction.png", "carbPrediction.png")

        print("Generating presigned URL...")
        prediction_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": f"{filename}_prediction.png"},
            ExpiresIn=3600,
        )

        print(f"Success! Loaded prediction url: {prediction_url}")

    return {"prediction_url": prediction_url}


create_bucket(bucket_name)

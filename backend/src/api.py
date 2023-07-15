from botocore.exceptions import ClientError

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as py

from .models import ProcessedDataResponse

import boto3
import datetime
import io


s3 = boto3.client("s3", region_name="us-east-1")
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")

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
    data_grouped = df.groupby('Minutes')['Glucose'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()

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
                s3.upload_fileobj(f, bucket_name, file_name, ExtraArgs={'ContentType': 'text/html'})
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
            return ProcessedDataResponse(message="Succes: File uploaded.", file=file.filename)
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
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Check if the image exists in S3
    try:
        s3_client.head_object(Bucket=bucket_name, Key=f"{filename}_line_plot.html")
        line_plot_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": f"{filename}_line_plot.html"},
            ExpiresIn=3600,
        )
    except ClientError:
        # Need to get data from S3 and then read the data from the object Body before
        # converting to pandas dataframe
        data = s3_client.get_object(Bucket=bucket_name, Key=f"{filename}")
        df = pd.read_csv(io.BytesIO(data["Body"].read()))

        generate_plot(df, "line_plot.html")
        upload_file_to_s3(bucket_name, f"{filename}_line_plot.html", "line_plot.html")

        line_plot_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": f"{filename}_line_plot.html"},
            ExpiresIn=3600,
        )

    return {"line_plot_url": line_plot_url}


bucket_name = "cgm-analytics-ucb"
create_bucket(bucket_name)

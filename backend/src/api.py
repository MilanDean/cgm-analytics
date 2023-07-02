from botocore.exceptions import ClientError

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pyathena import connect

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import seaborn as sns

from .models import ProcessedDataResponse

import boto3
import os
import time


s3 = boto3.client("s3", region_name="us-east-1")
glue = boto3.client('glue', region_name='us-east-1')
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
conn = connect(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
               aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
               s3_staging_dir='s3://cgm-analytics-ucb/',
               region_name='us-east-1')

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_plot(df, x_column=None, y_column=None, filename=None):

    font_path = font_manager.findfont(font_manager.FontProperties(family="Arial"))
    plt.rcParams["font.family"] = font_manager.get_font(font_path).family_name

    df['timestamp'] = pd.to_datetime(df['time'])

    plt.figure(figsize = (15,5))
    sns.lineplot(data=df, x=x_column, y=y_column, legend='brief', label=y_column)
    plt.title('Raw CGM Data')
    plt.tight_layout()

    plt.savefig(filename)


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
        with open(file_data, "rb") as file_obj:
            s3.upload_fileobj(file_obj, bucket_name, file_name)
            print(f"File {file_name} uploaded successfully to bucket {bucket_name}")
    except ClientError as e:
        print(e)
        return None
    

def check_table_exists(database: str, table: str):
    """Check if a table exists in a specified Glue database"""
    try:
        response = glue.get_table(DatabaseName=database, Name=table)
        return True
    except glue.exceptions.EntityNotFoundException:
        return False
    

def check_crawler_status(crawler_name: str):
    """Check if a Glue Crawler has completed its run"""
    while True:
        print("Waiting for AWS crawler to finish data schema preparation...")
        metrics = glue.get_crawler_metrics(CrawlerNameList=[crawler_name])
        if metrics['CrawlerMetricsList'][0]['StillEstimating'] == False:
            break
        time.sleep(10)
    return


@app.get("/")
async def root():
    # 501 error is the default `Not Implemented` status code
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/api/analysis", response_model=ProcessedDataResponse)
async def process_csv_file(file: UploadFile):
    try:
        # Check if file already exists in S3
        try:
            s3.head_object(Bucket=bucket_name, Key=file.filename)
            return ProcessedDataResponse(message="File already exists", file=file.filename)
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise

        # Upload CSV to S3
        file.file.seek(0)
        upload_file_to_s3(bucket_name, file.filename, file.file)

        # Refresh Athena table to run AWS Glue crawler only if the table doesn't exist
        if not check_table_exists("cgm-source-database", "cgm_analytics_ucb"):
            glue.start_crawler(Name='cgm-analytics-crawler')
            check_crawler_status('cgm-analytics-crawler')

        return ProcessedDataResponse(message="Success", file=file.filename)

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
    s3_client = boto3.client('s3', region_name='us-east-1')

    # Check if the file exists in S3
    try:
        s3_client.head_object(Bucket=bucket_name, Key=f'{filename}_line_plot.png')
        line_plot_url = s3_client.generate_presigned_url('get_object',
                                    Params={'Bucket': bucket_name, 'Key': f'{filename}_line_plot.png'},
                                    ExpiresIn=3600)
    except ClientError as e:
        # File does not exist, run the query and generate the graph
        query = """
                SELECT time, CGM 
                FROM "cgm-source-database"."cgm_analytics_ucb"
                """
        cursor = conn.cursor()
        cursor.execute(query)
        df = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(df, columns=columns)

        generate_plot(df, 'timestamp', 'CGM', 'line_plot.png')
        upload_file_to_s3(bucket_name, f'{filename}_line_plot.png', 'line_plot.png')

        line_plot_url = s3_client.generate_presigned_url('get_object',
                                    Params={'Bucket': bucket_name, 'Key': f'{filename}_line_plot.png'},
                                    ExpiresIn=3600)

    return {"line_plot_url": line_plot_url}


bucket_name = "cgm-analytics-ucb"
create_bucket(bucket_name)

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import io
import os
import uuid

app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Not optimal, in-memory data storage solution for datasets. Should work for MVP,
# but we dont want to rely on this long term
data_store = {}
graph_store = {}

@app.get("/")
async def root():
    # 501 error is the default `Not Implemented` status code
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/api/analysis")
async def process_csv_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        records = df.to_dict(orient="records")
        data_store['analysis_data'] = records

        return {"message": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis")
async def get_analysis_data():
    if "analysis_data" not in data_store:
        raise HTTPException(status_code=404, detail="Analysis data not available")

    return data_store["analysis_data"]


@app.get("/api/graph/{graph_id}")
async def get_graph(graph_id: str):
    if graph_id not in graph_store:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph_path = graph_store[graph_id]
    return FileResponse(graph_path, media_type="image/jpeg")


@app.post("/api/generate_graph")
async def generate_graph():
    if "analysis_data" not in data_store:
        raise HTTPException(status_code=404, detail="Analysis data not available")

    df = pd.DataFrame(data_store["analysis_data"])

    # Generate graphs based on your requirements
    graph1 = generate_graph_1(df)
    graph2 = generate_graph_2(df)

    # Save the graph images to byte buffers
    image_buffer1 = io.BytesIO()
    graph1.savefig(image_buffer1, format='jpeg')
    image_buffer1.seek(0)
    image_bytes1 = image_buffer1.getvalue()

    image_buffer2 = io.BytesIO()
    graph2.savefig(image_buffer2, format='jpeg')
    image_buffer2.seek(0)
    image_bytes2 = image_buffer2.getvalue()

    # Generate unique IDs for each graph
    graph_id1 = generate_unique_id()
    graph_id2 = generate_unique_id()

    # Create the graphs directory if it doesn't exist
    graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Store the image bytes in graph_store with their respective IDs
    graph_path1 = os.path.join(graphs_dir, f"{graph_id1}.jpeg")
    graph_path2 = os.path.join(graphs_dir, f"{graph_id2}.jpeg")

    with open(graph_path1, "wb") as f:
        f.write(image_bytes1)
    with open(graph_path2, "wb") as f:
        f.write(image_bytes2)

    graph_store[graph_id1] = graph_path1
    graph_store[graph_id2] = graph_path2

    return {"graph_ids": [graph_id1, graph_id2]}


def generate_graph_1(df):
    # Generate the first graph - Line Graph
    fig,ax = plt.subplots(figsize=(10,6))

    sns.lineplot(data=df, x='work_year', y='salary_in_usd', ax=ax, color='black')
    plt.locator_params(axis='x', nbins=4)
    ax.set_title("Data Science Salary by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Salary (USD)")

    return fig


def generate_graph_2(df):
    # Generate the second graph - Bar Graph
    fig,ax = plt.subplots(figsize=(10,6))

    sns.barplot(data=df.sort_values(by="salary_in_usd", ascending=False).head(10), x='job_title', y='salary_in_usd', ax=ax, palette='coolwarm')
    ax.set_title("Top 10 Salaries and Role")
    ax.set_xlabel("Role")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Salary (USD)")

    return fig


def generate_unique_id():
    # Generate a unique ID for the graphs
    return str(uuid.uuid4())

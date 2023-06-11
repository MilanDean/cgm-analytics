from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

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

@app.get("/")
async def root():
    # 501 error is the default `Not Implemented` status code
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/api/analysis")
async def process_csv_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        records = df.to_dict(orient="records")
        data_store['analysis_data'] = JSONResponse(content=records)

        return JSONResponse(content=records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis")
async def get_analysis_data():
    if "analysis_data" not in data_store:
        raise HTTPException(status_code=404, detail="Analysis data not available")
    
    print(data_store['analysis_data'])
    return data_store["analysis_data"]

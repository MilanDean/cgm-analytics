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


@app.get("/")
async def root():
    # 501 error is the default `Not Implemented` status code
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/api/analysis")
async def process_csv_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")


@app.get("/")
async def root():
    # 501 error is the default `Not Implemented` status code
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/process-file")
async def process_csv_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        # Return the modified data as JSON
        return JSONResponse(content=df.to_json(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
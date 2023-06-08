
from fastapi import FastAPI, HTTPException

app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")


@app.get("/hello")
async def table():
    pass

@app.get("/")
async def root():
    # 501 error is the default `Not Implemented` status code
    raise HTTPException(status_code=501, detail="Not implemented")
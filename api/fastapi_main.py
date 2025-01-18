from fastapi import FastAPI
from api.query import search_files

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/query")
async def query_files(query_string: str):

    result = search_files(query_string)

    print(result)
    return {"files": result}

import uvicorn

def main():
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8008, reload=True)

if __name__ == "__main__":
    main()
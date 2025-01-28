# inseek

## Installation and running

```
poetry env use 3.12
poetry install
poetry run uvicorn api.fastapi_main:app --reload
```

## Re-indexing Dev Method

```
poetry run python api/indexer.py
```

To index:

```
poetry run python api/test_thing.py --index DIRECTORY
```

To query:

```
poetry run python api/test_thing.py
```

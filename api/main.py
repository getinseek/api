from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from api.utils import search_images
from api.watcher import FileWatcher
import os
import mimetypes
import threading

app = FastAPI()
file_watcher = None

def start_file_watcher():
    global file_watcher

    # Watch the user's home directory
    watch_dir = os.path.expanduser("/Users/shubhampatil/Documents/programming/inseek/api/test")
    file_watcher = FileWatcher(watch_dir)
    
    try:
        file_watcher.watch()
    except Exception as e:
        print(f"Error in file watcher: {str(e)}")
        if file_watcher:
            file_watcher.stop()

def start():
    # Start the file watcher in a separate thread
    watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
    watcher_thread.start()
    
    print("Starting FastAPI server...")
    # Start the FastAPI server
    uvicorn.run("api.main:app", host="0.0.0.0", port=8008, reload=True)

@app.on_event("shutdown")
async def shutdown_event():
    global file_watcher
    if file_watcher:
        file_watcher.stop()

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/search", response_class=HTMLResponse)
async def search_page():
    return search_html

@app.get("/query")
async def query_files(query_string: str):
    result = search_images(query_string)
    return result  # Already in the correct format: {"files": [...]}

@app.get("/api/search")
async def search_endpoint(q: str):
    try:
        results = search_images(q)
        formatted_results = []
        
        if results["metadatas"]:  # Check if we have any results
            for metadata, distance in zip(results["metadatas"], results["distances"]):
                formatted_results.append({
                    "file_path": metadata["file_path"],
                    "file_name": metadata.get("file_name", os.path.basename(metadata["file_path"])),
                    "distance": float(distance)
                })
        
        return formatted_results
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return []

@app.get("/api/image")
async def get_image(path: str):
    """Serve image files."""
    if os.path.exists(path) and mimetypes.guess_type(path)[0].startswith('image'):
        return FileResponse(path)
    return {"error": "Image not found"}

# Store the HTML as a variable
search_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-container {
            margin-bottom: 20px;
        }
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
        }
        .result-item {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .result-item img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <input type="text" id="searchInput" placeholder="Search images...">
        <button onclick="search()">Search</button>
    </div>
    <div id="results" class="results"></div>

    <script>
        async function search() {
            const query = document.getElementById('searchInput').value;
            const response = await fetch(`/query?query_string=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            data.files.forEach(file => {
                const div = document.createElement('div');
                div.className = 'result-item';
                
                const img = document.createElement('img');
                img.src = `/image?path=${encodeURIComponent(file.file_path)}`;
                div.appendChild(img);
                
                const p = document.createElement('p');
                p.textContent = `${file.file_name} (Score: ${(1 - file.distance).toFixed(3)})`;
                div.appendChild(p);
                
                resultsDiv.appendChild(div);
            });
        }
    </script>
</body>
</html>
'''
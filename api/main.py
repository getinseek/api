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
    watch_dir = os.path.expanduser("~")
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
    
    # Format the results to match the expected structure
    formatted_results = []
    if result["metadatas"]:  # Check if we have any results
        for metadata, distance in zip(result["metadatas"], result["distances"]):
            formatted_results.append({
                "file_path": metadata["file_path"],
                "file_name": metadata.get("file_name", os.path.basename(metadata["file_path"])),
                "distance": float(distance)
            })
    
    return {"files": formatted_results}

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
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div x-data="searchApp()" class="container mx-auto px-4 py-8">
        <div class="max-w-5xl mx-auto">
            <h1 class="text-3xl font-bold text-center mb-8">Image Search</h1>
            
            <!-- Search Input -->
            <div class="mb-8">
                <input
                    type="text"
                    x-model="searchQuery"
                    @input.debounce.300ms="performSearch()"
                    placeholder="Search images..."
                    class="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
            </div>

            <!-- Loading State -->
            <div x-show="isLoading" class="text-center py-8">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-300 border-t-blue-500"></div>
            </div>

            <!-- Error State -->
            <div x-show="error" class="text-center py-8 text-red-500" x-text="error"></div>

            <!-- Results Grid -->
            <div x-show="!isLoading && results.length > 0" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <template x-for="result in results" :key="result.file_path">
                    <div class="bg-white rounded-lg shadow-md overflow-hidden">
                        <img :src="result.image_url" class="w-full h-48 object-cover">
                        <div class="p-4">
                            <div class="flex items-start justify-between">
                                <div class="flex-1 min-w-0">
                                    <p class="text-sm font-medium text-gray-900 truncate" x-text="result.file_name"></p>
                                    <p class="text-sm text-gray-500 truncate" x-text="result.file_path"></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </template>
            </div>

            <!-- No Results -->
            <div x-show="!isLoading && results.length === 0 && !error" class="text-center py-8 text-gray-500">
                No results found
            </div>
        </div>
    </div>

    <script>
        function searchApp() {
            return {
                searchQuery: '',
                results: [],
                isLoading: false,
                error: null,

                async performSearch() {
                    if (!this.searchQuery.trim()) {
                        this.results = [];
                        return;
                    }

                    this.isLoading = true;
                    this.error = null;

                    try {
                        const response = await fetch(`/api/search?q=${encodeURIComponent(this.searchQuery)}`);
                        const data = await response.json();

                        if (data.error) {
                            this.error = data.error;
                            this.results = [];
                        } else {
                            this.results = data.results;
                        }
                    } catch (err) {
                        this.error = 'An error occurred while searching';
                        this.results = [];
                    } finally {
                        this.isLoading = false;
                    }
                }
            };
        }
    </script>
</body>
</html>
'''
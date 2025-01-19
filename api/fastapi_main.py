from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api.query import search_files
import os

app = FastAPI()



@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/search", response_class=HTMLResponse)
async def search_page():
    return search_html

@app.get("/query")
async def query_files(query_string: str):

    result = search_files(query_string)

    print(result)
    return {"files": result}

@app.get("/api/search")
async def search_endpoint(q: str):
    try:
        results = search_files(q)
        formatted_results = []
        
        for result in results:
            dimensions = f"{result.metadata.get('width', 'N/A')}x{result.metadata.get('height', 'N/A')}" if result.metadata.get('width') else "N/A"
            
            formatted_results.append({
                "file_path": result.file_path,
                "file_name": result.metadata.get("file_name", "Unknown"),
                "description": result.metadata.get("text", "No description available"),
                "dimensions": dimensions,
                "score": f"{result.relevance_score:.2f}",
                "image_url": f"/api/image?path={result.file_path}"
            })
            
        return {"results": formatted_results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/image")
async def get_image(path: str):
    """Serve image files."""
    if os.path.exists(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return FileResponse(path)
    return {"error": "Image not found"}

# Store the HTML as a variable
search_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Search</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/alpinejs/3.13.5/cdn.min.js" defer></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.js"></script>
    <style>
        .image-container {
            width: 300px;
            height: 300px;
            position: relative;
            overflow: hidden;
        }
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center;
        }
        .hover-zoom {
            transition: transform 0.3s ease;
        }
        .hover-zoom:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div x-data="searchApp()" class="container mx-auto px-4 py-8">
        <div class="max-w-5xl mx-auto">
            <h1 class="text-3xl font-bold text-center mb-8">Image Search</h1>
            
            <!-- Search Form -->
            <div class="mb-8">
                <div class="flex gap-4">
                    <input 
                        type="text" 
                        x-model="searchQuery" 
                        @keyup.enter="performSearch"
                        placeholder="Enter search term..."
                        class="flex-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                    >
                    <button 
                        @click="performSearch"
                        class="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                        :disabled="isLoading"
                    >
                        <span x-show="!isLoading">Search</span>
                        <span x-show="isLoading">Searching...</span>
                    </button>
                </div>
            </div>

            <!-- Loading State -->
            <div x-show="isLoading" class="text-center py-8">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            </div>

            <!-- Error Message -->
            <div x-show="error" class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                <span class="block sm:inline" x-text="error"></span>
            </div>

            <!-- Results -->
            <div x-show="results.length > 0" class="space-y-6">
                <template x-for="result in results" :key="result.file_path">
                    <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                        <div class="flex flex-col md:flex-row gap-6">
                            <!-- Image Container with fixed dimensions -->
                            <div class="image-container flex-shrink-0 rounded-lg shadow-sm overflow-hidden">
                                <img 
                                    :src="result.image_url" 
                                    :alt="result.file_name"
                                    class="hover-zoom"
                                    @error="handleImageError($event)"
                                >
                            </div>
                            
                            <!-- Content Container -->
                            <div class="flex-1 min-w-0">
                                <h3 class="text-lg font-semibold truncate" x-text="result.file_name"></h3>
                                <p class="text-gray-600 mt-2 line-clamp-3" x-text="result.description"></p>
                                <div class="mt-3 grid grid-cols-2 gap-4 text-sm text-gray-500">
                                    <span x-text="'Dimensions: ' + result.dimensions"></span>
                                    <span x-text="'Score: ' + result.score"></span>
                                </div>
                                <div class="mt-2">
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

                handleImageError(event) {
                    event.target.src = 'https://via.placeholder.com/300x300?text=Image+Not+Found';
                },

                async performSearch() {
                    if (!this.searchQuery.trim()) return;
                    
                    this.isLoading = true;
                    this.error = null;
                    this.results = [];

                    try {
                        const response = await fetch(`/api/search?q=${encodeURIComponent(this.searchQuery)}`);
                        const data = await response.json();
                        
                        if (data.error) {
                            this.error = data.error;
                        } else {
                            this.results = data.results;
                        }
                    } catch (err) {
                        this.error = 'An error occurred while searching. Please try again.';
                    } finally {
                        this.isLoading = false;
                    }
                }
            }
        }
    </script>
</body>
</html>
'''
from api.indexer import index_files
from api.query import search_files

if __name__ == "__main__":
    print("Welcome to File Search")

    # # Prompt user for directory to index
    # try:
    #     repo_path = input(
    #         "Enter the path to the directory you want to index: ")
    #     if not repo_path:
    #         print("No directory specified. Exiting.")
    #         exit()

    #     # Index the specified directory
    #     print("Indexing files... This may take some time.")
    #     index_files(repo_path)
    #     print("Indexing complete. You can now search your files.")
    # except Exception as e:
    #     print(f"Error during indexing: {e}")
    #     exit()

    # Interactive query loop
    while True:
        try:
            query = input(
                "\nEnter your search query (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                print("Exiting File Search. Goodbye!")
                break

            # Perform the search
            results = search_files(query)

            # Display results
            if results:
                print("\nSearch Results:")
                for file_path in results:
                    print(f"- {file_path}")
            else:
                print("\nNo matching files found.")
        except Exception as e:
            print(f"Error during search: {e}")

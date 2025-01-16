import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from api.indexer import index_files
from api.query import search_files
from api.clear_index import clear_index


class FileSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Search Application")
        self.root.geometry("800x600")

        # Initialize panels
        self.home_panel = tk.Frame(root)
        self.query_panel = tk.Frame(root)

        self.setup_home_panel()
        self.setup_query_panel()

        # Start with Home Panel
        self.show_panel(self.home_panel)

    def setup_home_panel(self):
        """Sets up the Home Panel UI."""
        # Directory Selection
        tk.Label(self.home_panel, text="Selected Directory:").pack(pady=5)
        self.dir_entry = tk.Entry(self.home_panel, width=80)
        self.dir_entry.pack(pady=5)
        tk.Button(
            self.home_panel, text="Browse", command=self.browse_directory
        ).pack(pady=5)

        # Indexing Button
        self.index_button = tk.Button(
            self.home_panel, text="Index Directory", command=self.index_directory
        )
        self.index_button.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.home_panel, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)

        # Log Output
        tk.Label(self.home_panel, text="Logs:").pack(pady=5)
        self.log_text = tk.Text(self.home_panel, height=10, width=80)
        self.log_text.pack(pady=5)

        # Clear Index Button
        tk.Button(
            self.home_panel, text="Clear All Indexes", command=self.clear_indexes, bg="red", fg="red"
        ).pack(pady=5)

        # Make a Query Button (hidden by default)
        self.query_button = tk.Button(
            self.home_panel, text="Make a Query", command=lambda: self.show_panel(self.query_panel)
        )
        self.query_button.pack(pady=5)
        self.query_button.pack_forget()  # Initially hidden

    def setup_query_panel(self):
        """Sets up the Query Panel UI."""
        # Back Button
        tk.Button(
            self.query_panel, text="Back", command=lambda: self.show_panel(self.home_panel)
        ).pack(pady=5)

        # Search Query
        tk.Label(self.query_panel, text="Search Query:").pack(pady=5)
        self.query_entry = tk.Entry(self.query_panel, width=50)
        self.query_entry.pack(pady=5)
        tk.Button(
            self.query_panel, text="Search", command=self.search_files
        ).pack(pady=5)

        # Results Display
        self.results_frame = tk.Frame(self.query_panel)
        self.results_frame.pack(fill="both", expand=True)
        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar = ttk.Scrollbar(
            self.results_frame, orient="vertical", command=self.results_canvas.yview
        )
        self.scrollbar.pack(side="right", fill="y")
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.results_inner_frame = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_inner_frame, anchor="nw")
        self.results_inner_frame.bind("<Configure>", self.configure_scroll_region)

    def browse_directory(self):
        """Opens a file dialog to select a directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)

    def index_directory(self):
        """Indexes the selected directory."""
        directory = self.dir_entry.get()
        if not directory:
            messagebox.showerror("Error", "Please select a directory.")
            return

        self.log_text.insert(tk.END, "Indexing started...\n")
        self.progress["value"] = 0
        self.root.update()

        try:
            index_files(directory)
            self.log_text.insert(tk.END, "Indexing complete.\n")
            self.query_button.pack()  # Show the "Make a Query" button
        except Exception as e:
            self.log_text.insert(tk.END, f"Indexing failed: {e}\n")
            messagebox.showerror("Error", f"Indexing failed: {e}")

        self.progress["value"] = 100

    def search_files(self):
        """Searches for files based on the user's query."""
        query = self.query_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a search query.")
            return

        self.log_text.insert(tk.END, f"Searching for: {query}\n")
        try:
            results = search_files(query)
            if not results:
                self.log_text.insert(tk.END, "No results found.\n")
                return

            self.display_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {e}")

    def display_results(self, results):
        """Displays the search results in the GUI."""
        for widget in self.results_inner_frame.winfo_children():
            widget.destroy()

        for file_path in results:
            try:
                img = Image.open(file_path)
                img.thumbnail((200, 200))
                img_tk = ImageTk.PhotoImage(img)
                panel = tk.Label(self.results_inner_frame, image=img_tk)
                panel.image = img_tk
                panel.pack(pady=10)
            except Exception as e:
                self.log_text.insert(tk.END, f"Failed to display {file_path}: {e}\n")

    def clear_indexes(self):
        """Clears all indexes using the clear_index.py logic."""
        try:
            clear_index()
            self.log_text.insert(tk.END, "Indexes cleared successfully.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear indexes: {e}")

    def configure_scroll_region(self, event):
        """Configures the scroll region for the results canvas."""
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def show_panel(self, panel):
        """Displays the specified panel and hides others."""
        for widget in self.root.winfo_children():
            widget.pack_forget()
        panel.pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = FileSearchApp(root)
    root.mainloop()

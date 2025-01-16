import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from indexer import index_files
from query import search_files
from clear_index import clear_index


class ArgusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Argus")
        self.root.geometry("1200x800")  # Increased size
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Panels
        self.home_panel = ctk.CTkFrame(root)
        self.query_panel = ctk.CTkFrame(root)

        self.setup_home_panel()
        self.setup_query_panel()

        # Start with Home Panel
        self.show_panel(self.home_panel)

    def setup_home_panel(self):
        """Sets up the Home Panel UI."""
        title = ctk.CTkLabel(self.home_panel, text="Argus", font=("Arial", 28, "bold"))
        title.pack(pady=20)

        # Directory Selection
        self.dir_entry = ctk.CTkEntry(self.home_panel, placeholder_text="Select a directory to index", width=700)
        self.dir_entry.pack(pady=10)
        ctk.CTkButton(
            self.home_panel, text="Browse", command=self.browse_directory, width=150
        ).pack(pady=5)

        # Indexing Button
        ctk.CTkButton(
            self.home_panel, text="Index Directory", command=self.index_directory, width=200
        ).pack(pady=10)

        # Clear Index Button
        ctk.CTkButton(
            self.home_panel, text="Clear All Indexes", command=self.clear_indexes, fg_color="red", width=200
        ).pack(pady=5)

        # Progress Bar
        self.progress = ctk.CTkProgressBar(self.home_panel, orientation="horizontal", width=500, mode="determinate")
        self.progress.pack(pady=10)

        # Debug Panel (Resized to fit better)
        self.log_text = ctk.CTkTextbox(self.home_panel, height=250, width=900)
        self.log_text.pack(pady=10)

        # Make a Query Button (Initially hidden)
        self.query_button = ctk.CTkButton(
            self.home_panel, text="Make a Query", command=lambda: self.show_panel(self.query_panel), width=200
        )
        self.query_button.pack(pady=10)
        self.query_button.pack_forget()  # Initially hidden

    def setup_query_panel(self):
        """Sets up the Query Panel UI."""

        # Top Row: Back Button, Search Entry, Search Button
        top_row = ctk.CTkFrame(self.query_panel)
        top_row.pack(fill="x", pady=10, padx=10)

        # Back Button
        ctk.CTkButton(
            top_row, text="Back", command=lambda: self.show_panel(self.home_panel), width=100
        ).pack(side="left", padx=5)

        # Search Query
        self.query_entry = ctk.CTkEntry(top_row, placeholder_text="Enter search query", width=700)
        self.query_entry.pack(side="left", padx=5)

        # Search Button
        ctk.CTkButton(
            top_row, text="Search", command=self.search_files, width=100
        ).pack(side="left", padx=5)

        # Results Display
        self.results_frame = ctk.CTkScrollableFrame(self.query_panel, width=1000, height=600)
        self.results_frame.pack(pady=20, padx=10, fill="both", expand=True)

    def browse_directory(self):
        """Opens a file dialog to select a directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, directory)

    def index_directory(self):
        """Indexes the selected directory."""
        directory = self.dir_entry.get()
        if not directory:
            messagebox.showerror("Error", "Please select a directory.")
            return

        self.log_text.insert("end", "Indexing started...\n")
        self.progress.set(0.0)
        self.root.update()

        try:
            index_files(directory)
            self.log_text.insert("end", "Indexing complete.\n")
            self.query_button.pack()  # Show the "Make a Query" button
        except Exception as e:
            self.log_text.insert("end", f"Indexing failed: {e}\n")
            messagebox.showerror("Error", f"Indexing failed: {e}")

        self.progress.set(1.0)

    def search_files(self):
        """Searches for files based on the user's query."""
        query = self.query_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a search query.")
            return

        try:
            results = search_files(query)
            if not results:
                messagebox.showinfo("No Results", "No results found.")
                return

            self.display_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {e}")

    def display_results(self, results):
        """Displays the search results in the GUI."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for file_path in results:
            try:
                # Create a frame for each result
                result_frame = ctk.CTkFrame(self.results_frame, corner_radius=10, width=300, height=200)
                result_frame.pack(pady=10, padx=10, side="left", fill="both")

                # Display image
                img = Image.open(file_path)
                img.thumbnail((150, 150))
                img_tk = ImageTk.PhotoImage(img)
                img_label = ctk.CTkLabel(result_frame, image=img_tk)
                img_label.image = img_tk
                img_label.pack(pady=10)

                # Display metadata
                file_name = os.path.basename(file_path)
                metadata_label = ctk.CTkLabel(result_frame, text=file_name, wraplength=180)
                metadata_label.pack(pady=5)

            except Exception as e:
                self.log_text.insert("end", f"Failed to display {file_path}: {e}\n")

    def clear_indexes(self):
        """Clears all indexes using the clear_index.py logic."""
        try:
            clear_index()
            self.log_text.insert("end", "Indexes cleared successfully.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear indexes: {e}")

    def show_panel(self, panel):
        """Displays the specified panel and hides others."""
        for widget in self.root.winfo_children():
            widget.pack_forget()
        panel.pack(fill="both", expand=True)


if __name__ == "__main__":
    root = ctk.CTk()
    app = ArgusApp(root)
    root.mainloop()




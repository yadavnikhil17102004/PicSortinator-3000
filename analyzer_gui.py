"""
GUI Interface for Screenshot Analyzer
Enhanced Version with Google Cloud Vision integration
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from PIL import Image, ImageTk
from screenshot_analyzer import ScreenshotAnalyzer, Category

class ScreenshotAnalyzerGUI:
    """GUI application for Screenshot Analyzer."""
    
    def __init__(self, root, credentials_path=None):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window.
            credentials_path: Path to Google Cloud credentials file.
        """
        self.root = root
        self.root.title("Screenshot Analyzer for Digital Investigations")
        self.root.geometry("1000x700")
        
        self.analyzer = ScreenshotAnalyzer(credentials_path=credentials_path)
        self.current_image_path = None
        self.search_results = []
        self.current_result_index = 0
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Image Analysis")
        
        # Search tab
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Search & Export")
        
        # Setup Analysis tab
        self._setup_analysis_tab()
        
        # Setup Search tab
        self._setup_search_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_analysis_tab(self):
        """Setup the Analysis tab."""
        # Frame for buttons
        button_frame = ttk.Frame(self.analysis_frame)
        button_frame.pack(pady=10, fill=tk.X)
        
        # Load Image button
        load_button = ttk.Button(button_frame, text="Load Image", command=self._load_image)
        load_button.pack(side=tk.LEFT, padx=10)
        
        # Analyze button
        analyze_button = ttk.Button(button_frame, text="Analyze Image", command=self._analyze_image)
        analyze_button.pack(side=tk.LEFT, padx=10)
        
        # Save Results button
        save_button = ttk.Button(button_frame, text="Save Results", command=self._save_results)
        save_button.pack(side=tk.LEFT, padx=10)
        
        # Frame for image and results
        content_frame = ttk.Frame(self.analysis_frame)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Image preview frame
        self.image_frame = ttk.LabelFrame(content_frame, text="Image Preview")
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(content_frame, text="Analysis Results")
        results_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Results notebook (tabbed results)
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # JSON tab
        json_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(json_frame, text="JSON")
        
        self.results_text = scrolledtext.ScrolledText(json_frame, wrap=tk.WORD)
        self.results_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # OCR Text tab
        ocr_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(ocr_frame, text="OCR Text")
        
        self.ocr_text = scrolledtext.ScrolledText(ocr_frame, wrap=tk.WORD)
        self.ocr_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Tags tab
        tags_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(tags_frame, text="Tags")
        
        self.tags_tree = ttk.Treeview(tags_frame, columns=("Tag", "Type", "Confidence"), show="headings")
        self.tags_tree.heading("Tag", text="Tag")
        self.tags_tree.heading("Type", text="Type")
        self.tags_tree.heading("Confidence", text="Confidence")
        self.tags_tree.column("Tag", width=300)
        self.tags_tree.column("Type", width=100)
        self.tags_tree.column("Confidence", width=100)
        self.tags_tree.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
    
    def _setup_search_tab(self):
        """Setup the Search tab."""
        # Search controls frame
        search_controls = ttk.LabelFrame(self.search_frame, text="Search Criteria")
        search_controls.pack(fill=tk.X, padx=10, pady=10)
        
        # Search query
        ttk.Label(search_controls, text="Text Query:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.search_query_var = tk.StringVar()
        ttk.Entry(search_controls, textvariable=self.search_query_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Category dropdown
        ttk.Label(search_controls, text="Category:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.category_var = tk.StringVar()
        category_values = ["", "chat", "transaction", "threat", "adult_content", "uncategorized"]
        ttk.Combobox(search_controls, textvariable=self.category_var, values=category_values, width=15).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Tags entry
        ttk.Label(search_controls, text="Tags (comma separated):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.tags_var = tk.StringVar()
        ttk.Entry(search_controls, textvariable=self.tags_var, width=30).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Search button
        ttk.Button(search_controls, text="Search", command=self._search_images).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Export controls
        export_frame = ttk.Frame(search_controls)
        export_frame.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        self.export_format_var = tk.StringVar(value="json")
        ttk.Radiobutton(export_frame, text="JSON", variable=self.export_format_var, value="json").pack(side=tk.LEFT)
        ttk.Radiobutton(export_frame, text="CSV", variable=self.export_format_var, value="csv").pack(side=tk.LEFT)
        ttk.Radiobutton(export_frame, text="PDF", variable=self.export_format_var, value="pdf").pack(side=tk.LEFT)
        
        ttk.Button(search_controls, text="Export", command=self._export_results).grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
        
        # Results list frame
        results_frame = ttk.LabelFrame(self.search_frame, text="Search Results")
        results_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Results treeview
        columns = ("ID", "Category", "Detected Text", "Tags")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
        
        self.results_tree.heading("ID", text="ID")
        self.results_tree.heading("Category", text="Category")
        self.results_tree.heading("Detected Text", text="Detected Text")
        self.results_tree.heading("Tags", text="Tags")
        
        self.results_tree.column("ID", width=80)
        self.results_tree.column("Category", width=100)
        self.results_tree.column("Detected Text", width=400)
        self.results_tree.column("Tags", width=300)
        
        self.results_tree.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind double-click to view result
        self.results_tree.bind("<Double-1>", self._view_search_result)
    
    def _load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            
            # Display image preview
            self._display_image(file_path)
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.ocr_text.delete(1.0, tk.END)
            self.tags_tree.delete(*self.tags_tree.get_children())
    
    def _display_image(self, file_path):
        """
        Display image preview.
        
        Args:
            file_path (str): Path to the image file.
        """
        try:
            # Open and resize image for preview
            img = Image.open(file_path)
            
            # Calculate new dimensions while maintaining aspect ratio
            max_width = 400
            max_height = 400
            
            width, height = img.size
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def _analyze_image(self):
        """Analyze the loaded image."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Analyzing image...")
            self.root.update()
            
            # Process the image
            result = self.analyzer.process_image(self.current_image_path)
            
            # Display JSON results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, json.dumps(result, indent=2))
            
            # Display OCR text
            self.ocr_text.delete(1.0, tk.END)
            if result.get("detected_text"):
                self.ocr_text.insert(tk.END, result.get("detected_text"))
            else:
                self.ocr_text.insert(tk.END, "No text detected.")
            
            # Display tags
            self.tags_tree.delete(*self.tags_tree.get_children())
            for tag in result.get("tags", []):
                # For simplicity, split tag by dots to get type
                parts = tag.split(".")
                tag_type = parts[0] if len(parts) > 0 else "unknown"
                tag_value = ".".join(parts[1:]) if len(parts) > 1 else tag
                self.tags_tree.insert("", tk.END, values=(tag_value, tag_type, ""))
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")
    
    def _save_results(self):
        """Save analysis results to a file."""
        if not self.results_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                
                self.status_var.set(f"Results saved to {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def _search_images(self):
        """Search for images based on criteria."""
        try:
            self.status_var.set("Searching...")
            self.root.update()
            
            # Get search criteria
            query = self.search_query_var.get().strip()
            category = self.category_var.get().strip()
            tags_str = self.tags_var.get().strip()
            tags = [t.strip() for t in tags_str.split(',')] if tags_str else None
            
            # Search for images
            results = self.analyzer.search_images(query, tags, category)
            self.search_results = results
            
            # Clear results tree
            self.results_tree.delete(*self.results_tree.get_children())
            
            # Populate results tree
            for result in results:
                # Truncate text for display
                text_preview = result.get("detected_text", "")[:50] + "..." if len(result.get("detected_text", "")) > 50 else result.get("detected_text", "")
                # Join tags for display
                tags_preview = ", ".join(result.get("tags", [])[:3]) + "..." if len(result.get("tags", [])) > 3 else ", ".join(result.get("tags", []))
                
                self.results_tree.insert("", tk.END, values=(
                    result.get("image_id", "")[:8],
                    result.get("category", ""),
                    text_preview,
                    tags_preview
                ))
            
            self.status_var.set(f"Found {len(results)} results")
            
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            self.status_var.set("Search failed")
    
    def _view_search_result(self, event):
        """View details of a search result."""
        # Get selected item
        selected_item = self.results_tree.focus()
        if not selected_item:
            return
        
        # Get item index
        selected_index = self.results_tree.index(selected_item)
        if selected_index >= len(self.search_results):
            return
        
        # Get result
        result = self.search_results[selected_index]
        
        # Show result in dialog
        result_dialog = tk.Toplevel(self.root)
        result_dialog.title("Result Details")
        result_dialog.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(result_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # JSON tab
        json_frame = ttk.Frame(notebook)
        notebook.add(json_frame, text="JSON")
        
        json_text = scrolledtext.ScrolledText(json_frame, wrap=tk.WORD)
        json_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        json_text.insert(tk.END, json.dumps(result, indent=2))
        
        # OCR tab
        ocr_frame = ttk.Frame(notebook)
        notebook.add(ocr_frame, text="OCR Text")
        
        ocr_text = scrolledtext.ScrolledText(ocr_frame, wrap=tk.WORD)
        ocr_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ocr_text.insert(tk.END, result.get("detected_text", "No text detected."))
        
        # Tags tab
        tags_frame = ttk.Frame(notebook)
        notebook.add(tags_frame, text="Tags")
        
        tags_tree = ttk.Treeview(tags_frame, columns=("Tag", "Type"), show="headings")
        tags_tree.heading("Tag", text="Tag")
        tags_tree.heading("Type", text="Type")
        tags_tree.column("Tag", width=300)
        tags_tree.column("Type", width=100)
        tags_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Populate tags
        for tag in result.get("tags", []):
            parts = tag.split(".")
            tag_type = parts[0] if len(parts) > 0 else "unknown"
            tag_value = ".".join(parts[1:]) if len(parts) > 1 else tag
            tags_tree.insert("", tk.END, values=(tag_value, tag_type))
    
    def _export_results(self):
        """Export search results."""
        if not self.search_results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        format_type = self.export_format_var.get()
        file_path = filedialog.asksaveasfilename(
            title=f"Export Results as {format_type.upper()}",
            defaultextension=f".{format_type}",
            filetypes=[(f"{format_type.upper()} Files", f"*.{format_type}"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Exporting to {format_type}...")
                self.root.update()
                
                # Export results
                exported_file = self.analyzer.export_results(self.search_results, format_type, file_path)
                
                self.status_var.set(f"Exported to {os.path.basename(exported_file)}")
                messagebox.showinfo("Export Complete", f"Results exported to {exported_file}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
                self.status_var.set("Export failed")


def main():
    """Run the GUI application."""
    # Get credentials path from command line if provided
    credentials_path = None
    if len(sys.argv) > 1:
        credentials_path = sys.argv[1]
    
    root = tk.Tk()
    app = ScreenshotAnalyzerGUI(root, credentials_path)
    root.mainloop()


if __name__ == "__main__":
    main()

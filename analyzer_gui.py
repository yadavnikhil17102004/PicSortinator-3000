"""
GUI Interface for Screenshot Analyzer
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
from screenshot_analyzer import ScreenshotAnalyzer

class ScreenshotAnalyzerGUI:
    """GUI application for Screenshot Analyzer."""
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window.
        """
        self.root = root
        self.root.title("Screenshot Analyzer for Digital Investigations")
        self.root.geometry("800x600")
        
        self.analyzer = ScreenshotAnalyzer()
        self.current_image_path = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10, fill=tk.X)
        
        # Load Image button
        load_button = tk.Button(button_frame, text="Load Image", command=self._load_image)
        load_button.pack(side=tk.LEFT, padx=10)
        
        # Analyze button
        analyze_button = tk.Button(button_frame, text="Analyze Image", command=self._analyze_image)
        analyze_button.pack(side=tk.LEFT, padx=10)
        
        # Save Results button
        save_button = tk.Button(button_frame, text="Save Results", command=self._save_results)
        save_button.pack(side=tk.LEFT, padx=10)
        
        # Frame for image and results
        content_frame = tk.Frame(self.root)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Image preview frame
        self.image_frame = tk.LabelFrame(content_frame, text="Image Preview")
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Results frame
        results_frame = tk.LabelFrame(content_frame, text="Analysis Results")
        results_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
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
            max_width = 350
            max_height = 350
            
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
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, json.dumps(result, indent=2))
            
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


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = ScreenshotAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
CLI Interface for Screenshot Analyzer
"""

import os
import sys
import json
import argparse
from screenshot_analyzer import ScreenshotAnalyzer

def process_images(images, output_file=None):
    """
    Process multiple images and optionally save results to a file.
    
    Args:
        images (list): List of image file paths.
        output_file (str, optional): Path to save the results.
    """
    analyzer = ScreenshotAnalyzer()
    results = []
    
    for image_path in images:
        print(f"Processing: {image_path}")
        result = analyzer.process_image(image_path)
        results.append(result)
        
        # Print individual result
        print(json.dumps(result, indent=2))
        print("-" * 50)
    
    # Save results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


def main():
    """Parse command line arguments and process images."""
    parser = argparse.ArgumentParser(description="Screenshot Analyzer for Digital Investigations")
    
    parser.add_argument("images", nargs="+", help="Paths to image files")
    parser.add_argument("-o", "--output", help="Output file to save results")
    
    args = parser.parse_args()
    
    # Validate image paths
    valid_images = []
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"Warning: File not found - {image_path}")
        else:
            valid_images.append(image_path)
    
    if not valid_images:
        print("Error: No valid image files provided")
        sys.exit(1)
    
    # Process the images
    process_images(valid_images, args.output)


if __name__ == "__main__":
    main()

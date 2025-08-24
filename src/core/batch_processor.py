"""
Batch Processing Script for Screenshot Analyzer
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from screenshot_analyzer import ScreenshotAnalyzer

def process_directory(directory_path, output_dir=None, recursive=False):
    """
    Process all image files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing images.
        output_dir (str, optional): Directory to save individual JSON results.
        recursive (bool): Whether to process subdirectories recursively.
    
    Returns:
        list: List of processing results.
    """
    analyzer = ScreenshotAnalyzer()
    results = []
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all image files in the directory
    pattern = os.path.join(directory_path, '**' if recursive else '*')
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        if recursive:
            image_files.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
        else:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    # Process each image
    total_images = len(image_files)
    print(f"Found {total_images} image(s) to process")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{total_images}: {image_path}")
        
        try:
            # Process the image
            result = analyzer.process_image(image_path)
            
            # Add file path to result for reference
            result["file_path"] = image_path
            
            # Save individual result if output directory is specified
            if output_dir:
                base_name = os.path.basename(image_path)
                file_name = os.path.splitext(base_name)[0]
                json_path = os.path.join(output_dir, f"{file_name}_result.json")
                
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
            
            # Add to overall results
            results.append(result)
            
            # Print result status
            if result.get("processable"):
                print(f"  ✅ Processable: {result['image_id']}")
            else:
                print(f"  ❌ Not processable: {result['image_id']}")
                
        except Exception as e:
            print(f"  ❌ Error processing {image_path}: {str(e)}")
            
            # Add error to results
            results.append({
                "image_id": "error",
                "file_path": image_path,
                "processable": False,
                "error": str(e)
            })
    
    return results


def main():
    """Parse command line arguments and process directory."""
    parser = argparse.ArgumentParser(description="Batch process screenshots for digital investigations")
    
    parser.add_argument("directory", help="Directory containing images to process")
    parser.add_argument("-o", "--output", help="Directory to save individual JSON results")
    parser.add_argument("-s", "--summary", help="Path to save summary JSON file")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
    
    # Process the directory
    start_time = datetime.now()
    print(f"Starting batch processing at {start_time}")
    
    results = process_directory(args.directory, args.output, args.recursive)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Batch processing completed in {duration:.2f} seconds")
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "directory": args.directory,
        "recursive": args.recursive,
        "total_images": len(results),
        "processable_images": sum(1 for r in results if r.get("processable")),
        "results": results
    }
    
    # Save summary if requested
    if args.summary:
        with open(args.summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {args.summary}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total images processed: {summary['total_images']}")
    print(f"Processable images: {summary['processable_images']}")
    print(f"Non-processable images: {summary['total_images'] - summary['processable_images']}")


if __name__ == "__main__":
    main()

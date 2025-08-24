"""
Test Script for Screenshot Analyzer
"""

import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont
from screenshot_analyzer import ScreenshotAnalyzer

def create_test_image(output_path, with_text=True):
    """
    Create a test image for testing the analyzer.
    
    Args:
        output_path (str): Path to save the test image.
        with_text (bool): Whether to include text in the image.
    """
    # Create a blank image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some UI elements (rectangles)
    draw.rectangle([(50, 50), (750, 150)], outline='black', width=2)
    draw.rectangle([(50, 200), (750, 300)], outline='black', width=2)
    draw.rectangle([(50, 350), (750, 450)], outline='black', width=2)
    
    # Add some text if requested
    if with_text:
        try:
            # Try to get a font, fall back to default if not available
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((100, 100), "This is a test screenshot", fill='black', font=font)
        draw.text((100, 250), "For digital investigations", fill='black', font=font)
        draw.text((100, 400), "Classification system", fill='black', font=font)
    
    # Save the image
    img.save(output_path)
    print(f"Test image created: {output_path}")
    return output_path


def run_test():
    """Run tests on the Screenshot Analyzer."""
    analyzer = ScreenshotAnalyzer()
    
    # Create test directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create and analyze test images
    test_image_with_text = os.path.join(test_dir, "test_with_text.png")
    test_image_without_text = os.path.join(test_dir, "test_without_text.png")
    
    create_test_image(test_image_with_text, with_text=True)
    create_test_image(test_image_without_text, with_text=False)
    
    # Process the images
    print("\nAnalyzing image with text:")
    result1 = analyzer.process_image(test_image_with_text)
    print(json.dumps(result1, indent=2))
    
    print("\nAnalyzing image without text:")
    result2 = analyzer.process_image(test_image_without_text)
    print(json.dumps(result2, indent=2))
    
    # Verify results
    if result1["processable"] and not result1.get("error"):
        print("\n✅ Test passed: Successfully processed image with text")
    else:
        print("\n❌ Test failed: Could not process image with text")
    
    if result2["processable"] and not result2.get("error"):
        print("✅ Test passed: Successfully processed image without text")
    else:
        print("❌ Test failed: Could not process image without text")


if __name__ == "__main__":
    run_test()

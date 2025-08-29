#!/usr/bin/env python3
"""
Custom Tagging Example
======================
Demonstrates how to customize the tagging system with your own categories.
"""

import sys
import os

# Add the parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tagging import ImageTagger
from modules.model_manager import ModelManager

def custom_tagging_example():
    """Demonstrate custom tagging configuration."""
    
    print("🏷️  PicSortinator 3000 - Custom Tagging Example")
    print("=" * 60)
    
    # Initialize tagger
    tagger = ImageTagger()
    
    # Example: Custom categories for specific use cases
    custom_categories = {
        "photography": [
            "portrait", "landscape", "macro", "street", "nature",
            "wildlife", "architecture", "abstract", "black_and_white"
        ],
        "objects": [
            "camera", "lens", "tripod", "computer", "phone", "book",
            "car", "bicycle", "building", "tree", "flower", "animal"
        ],
        "events": [
            "wedding", "party", "graduation", "vacation", "holiday",
            "birthday", "concert", "sports", "meeting", "conference"
        ],
        "emotions": [
            "happy", "sad", "excited", "peaceful", "energetic",
            "romantic", "dramatic", "mysterious", "playful"
        ]
    }
    
    print("📁 Custom Categories:")
    for category, tags in custom_categories.items():
        print(f"  {category}: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
    
    # You can modify the tagger's useful_categories
    print(f"\n🔧 Current useful categories: {len(tagger.useful_categories)} keywords")
    
    # Add custom categories to the tagger
    all_custom_tags = []
    for category_tags in custom_categories.values():
        all_custom_tags.extend(category_tags)
    
    # Update tagger with additional categories
    original_count = len(tagger.useful_categories)
    tagger.useful_categories.update(all_custom_tags)
    new_count = len(tagger.useful_categories)
    
    print(f"✅ Added {new_count - original_count} new custom categories")
    print(f"📊 Total categories now: {new_count}")
    
    # Example: Process an image with custom categories
    test_image = "test_images/sample.jpg"  # Update this path
    
    if os.path.exists(test_image):
        print(f"\n🎯 Testing custom tagging on: {test_image}")
        
        # Get predictions with custom categories
        tags = tagger.tag_image(test_image)
        print(f"📋 Generated tags: {tags}")
        
        # Show which custom categories were detected
        detected_custom = []
        for tag in tags:
            for category, category_tags in custom_categories.items():
                if tag in category_tags:
                    detected_custom.append(f"{tag} ({category})")
        
        if detected_custom:
            print(f"🎨 Custom category matches: {', '.join(detected_custom)}")
        else:
            print("🔍 No custom category matches (try different images)")
    else:
        print(f"\n⚠️  Test image not found: {test_image}")
        print("Create a test_images directory with sample photos to test custom tagging")
    
    # Example: Confidence threshold adjustment
    print(f"\n⚙️  Current confidence threshold: {tagger.confidence_threshold}")
    print("💡 Tip: Lower threshold (0.1-0.3) = more tags, higher threshold (0.4-0.8) = fewer, more confident tags")
    
    # Example: Show model information
    model_manager = ModelManager()
    labels = model_manager.load_imagenet_labels()
    print(f"🧠 Using model with {len(labels)} total classes")
    
    print("\n🎉 Custom tagging configuration complete!")
    print("You can modify the useful_categories in modules/tagging.py for permanent changes")

if __name__ == "__main__":
    custom_tagging_example()

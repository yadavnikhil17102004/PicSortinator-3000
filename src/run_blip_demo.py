
import sys
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract

# Try to set tesseract_cmd to the default Windows install location if not already set
import shutil
if not pytesseract.pytesseract.tesseract_cmd or not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
	tesseract_path = shutil.which("tesseract")
	if tesseract_path:
		pytesseract.pytesseract.tesseract_cmd = tesseract_path
	else:
		# Try the default install location
		default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
		if os.path.isfile(default_path):
			pytesseract.pytesseract.tesseract_cmd = default_path

def load_image(image_path=None):
	if image_path and os.path.isfile(image_path):
		return Image.open(image_path).convert('RGB')
	else:
		print("No valid image path provided, using demo image from the web.")
		img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
		return Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

def main():
	# Load processor and model
	processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
	model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

	# Get image path from command-line arguments
	image_path = sys.argv[1] if len(sys.argv) > 1 else None
	raw_image = load_image(image_path)

	# Conditional image captioning
	text = "a photography "
	inputs = processor(raw_image, text, return_tensors="pt")
	out = model.generate(**inputs)
	print("Conditional caption:", processor.decode(out[0], skip_special_tokens=True))


	# OCR: Extract text from the image using pytesseract
	print("\n--- OCR (Tesseract) Output ---")
	try:
		if not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
			raise FileNotFoundError(f"Tesseract executable not found at '{pytesseract.pytesseract.tesseract_cmd}'. Please check your installation and PATH.")
		ocr_text = pytesseract.image_to_string(raw_image)
		print(ocr_text.strip() if ocr_text.strip() else "No text detected.")
	except Exception as e:
		print(f"OCR failed: {e}")

	# Conditional image captioning
	text = "a photography of and contain"
	inputs = processor(raw_image, text, return_tensors="pt")
	out = model.generate(**inputs)
	print("\nConditional caption:", processor.decode(out[0], skip_special_tokens=True))

	# Unconditional image captioning
	inputs = processor(raw_image, return_tensors="pt")
	out = model.generate(**inputs)
	print("Unconditional caption:", processor.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
	main()

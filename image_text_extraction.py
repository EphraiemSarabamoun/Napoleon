import easyocr

# Initialize EasyOCR Reader (English language)
reader = easyocr.Reader(['en'])

# Perform OCR on your image within the output_frames directory
results = reader.readtext('output_frames/frame_20250324-171934.jpg')

# Print extracted text
for bbox, text, confidence in results:
    print(f'Text: {text}, Confidence: {confidence:.2f}')
import cv2
import pytesseract
import numpy as np
import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer

# Load LayoutLM model and tokenizer
model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')
tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')

# Define threshold distance between columns
COLUMN_THRESHOLD = 50

# Load document image and preprocess it
image = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.medianBlur(image, 3)
image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Convert image to text using OCR
text = pytesseract.image_to_string(image)

# Tokenize text using LayoutLM tokenizer
tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')

# Get layout embeddings from LayoutLM model
with torch.no_grad():
    outputs = model(tokens)
    embeddings = outputs.last_hidden_state

# Get coordinates of each token in the document
positions = []
for i, token in enumerate(tokens[0]):
    position = embeddings[0][i][4:6].cpu().numpy()
    positions.append(position)

# Group tokens into columns based on their x-coordinates
columns = []
current_column = []
for i, position in enumerate(positions):
    if i == 0:
        current_column.append(i)
    else:
        if abs(position[0] - positions[i-1][0]) < COLUMN_THRESHOLD:
            current_column.append(i)
        else:
            columns.append(current_column)
            current_column = [i]
columns.append(current_column)

# Count the number of columns
num_columns = len(columns)
print(f'The document has {num_columns} columns.')

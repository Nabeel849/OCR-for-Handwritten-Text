import cv2
import pytesseract

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Resize the image to a larger size for better OCR accuracy
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding to binarize the image
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to connect disjoint characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

    return closed_image

def extract_text(image):
    # Extract text from the image using Tesseract with custom configurations
    custom_config = r'--oem 3 --psm 6'  # Default OCR Engine and Page Segmentation Mode
    extracted_text = pytesseract.image_to_string(image, config=custom_config)
    return extracted_text

def save_text_to_file(text, output_path):
    # Save the extracted text to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    # Specify the path to your image
    image_path = 'D:/Work/IPCV/OCR/handwrite-anything-for-you-in-neat-clear-handwriting-0a3b.jpg'  # Change this to your image path
    output_path = 'D:/Work/IPCV/OCR/extracted_text.txt'  # Path to save the extracted text

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Extract text from the processed image
    extracted_text = extract_text(processed_image)

    # Print the extracted text
    print("Extracted Text:")
    print(extracted_text)

    # Save the extracted text to a file
    save_text_to_file(extracted_text, output_path)
    print(f"Extracted text saved to {output_path}")

    # Optional: Display the original and processed images
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

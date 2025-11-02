import os
from pypdf import PdfReader

# Directory containing the source PDF documents
PDF_SOURCE_DIR = "medical_data"
# File to save the extracted text
OUTPUT_TEXT_FILE = "knowledge_base.txt"

def extract_text_from_pdfs(pdf_dir):
    """
    Extracts text from all PDF files in a given directory.
    
    Args:
        pdf_dir (str): The path to the directory containing PDF files.
    
    Returns:
        str: A single string containing all the extracted text.
    """
    all_text = ""
    print(f"Scanning for PDF files in '{pdf_dir}'...")
    if not os.path.exists(pdf_dir):
        print(f"Error: Directory '{pdf_dir}' not found. Please create it and add your medical PDFs.")
        return ""
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'.")
        return ""
    
    print(f"Found {len(pdf_files)} PDF(s). Extracting text...")
    
    for filename in pdf_files:
        path = os.path.join(pdf_dir, filename)
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n\n" # Add space between pages
            print(f" - Successfully extracted text from {filename}")
        except Exception as e:
            print(f" - Could not read {filename}: {e}")
    return all_text

def main():
    """
    Main function to run the data preparation process.
    """
    extracted_text = extract_text_from_pdfs(PDF_SOURCE_DIR)
    if extracted_text:
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"\nSuccessfully extracted and saved text to '{OUTPUT_TEXT_FILE}'.")
        print(f"Total characters extracted: {len(extracted_text)}")
    else:
        print("\nNo text was extracted. Exiting.")

if __name__ == "__main__":
    main()
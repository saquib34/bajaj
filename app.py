from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import re
import os
import shutil
from typing import List, Dict, Any
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="Lab Test OCR API", 
              description="Extract lab test information from medical report images using PaddleOCR")

# Initialize PaddleOCR
ocr = PaddleOCR(
        use_angle_cls=False,  # Disable angle detection to save memory
        lang='en',
        use_gpu=False,  
        enable_mkldnn=True, 
        det_db_thresh=0.3, 
        det_db_box_thresh=0.5, 
        rec_batch_num=1  
)
# Helper functions from your original code
def clean_text(text):
    return text.strip().replace(":", "").replace(")", "").replace("(", "").strip()

def is_result(text):
    return re.match(r'^\d+(\.\d+)?$', text)

def is_range(text):
    return re.match(r'^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$', text)

def group_rows(ocr_data, y_threshold=20):
    rows = []
    current_row = []
    prev_y = None

    # Ensure the OCR data is sorted by the y-center of the bounding boxes
    ocr_data = sorted(ocr_data, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)

    for box, (text, conf) in ocr_data:
        y_center = (box[0][1] + box[2][1]) / 2

        if prev_y is None or abs(y_center - prev_y) < y_threshold:
            current_row.append((box, text, conf))
        else:
            rows.append(current_row)
            current_row = [(box, text, conf)]

        prev_y = y_center

    # Append the last row if it exists
    if current_row:
        rows.append(current_row)

    return rows

def extract_tests_from_rows(rows):
    extracted = []
    for row in rows:
        name = None
        result = None
        ref_range = None
        unit = None

        for box, text, conf in row:
            text_clean = clean_text(text)

            if is_result(text_clean):
                result = text_clean
            elif is_range(text_clean):
                ref_range = text_clean
            elif len(text_clean) < 10 and any(c.isalpha() for c in text_clean) and result and not unit:
                # Assume this is unit if it's short, alphabetic, and after result
                unit = text_clean
            elif len(text_clean) > 2 and not is_result(text_clean) and not is_range(text_clean):
                name = text_clean if name is None else name + " " + text_clean

        if name and result and ref_range:
            extracted.append({
                'test_name': name[20:] if len(name) > 20 else name,
                'result': result,
                'unit': unit,
                'reference_range': ref_range
            })

    return extracted

def extract_lab_tests(ocr_data):
    rows = group_rows(ocr_data)
    tests = extract_tests_from_rows(rows)
    return tests

class TestResult(BaseModel):
    test_name: str
    result: str
    unit: str = None
    reference_range: str

class OCRResponse(BaseModel):
    tests: List[TestResult]
    image_name: str
    total_tests_found: int

@app.post("/get-lab-tests", response_model=OCRResponse)
async def extract_tests_from_image(file: UploadFile = File(...)):
    """
    Upload a medical report image and extract structured lab test information.
    Returns a list of test names, results, units, and reference ranges.
    """
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create a temporary file to save the uploaded image
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image with PaddleOCR
        result = ocr.ocr(temp_file_path, cls=True)
        
        # Convert OCR results to the format expected by extract_lab_tests
        list_result = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                list_result.append(line)
        
        # Extract lab tests from OCR results
        tests = extract_lab_tests(list_result)
        
        # Return the structured data
        return OCRResponse(
            tests=tests,
            image_name=file.filename,
            total_tests_found=len(tests)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/")
def read_root():
    return {"message": "Lab Test OCR API is running. Upload an image to /extract-lab-tests/ to extract lab test information."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

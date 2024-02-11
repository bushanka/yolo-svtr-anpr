# ANPR - Automobile Number Plate Recognition

This project is an API for recognizing automobile number plates, specifically designed for Russian car numbers. The API is built on FastAPI and uses YOLO (You Only Look Once) for object detection and SVTR for Optical Character Recognition (OCR).

## Requirements
- Python 3.6+
- FastAPI
- YOLO model
- SVTR OCR model

## Installation
1. Clone this repository.
2. Install the required dependencies by running:
    
    pip install -r requirements.txt
    
3. Download and place the YOLO model weights in the models directory.
4. Download and place the SVTR OCR model in the models directory.

Using Docker:
## Build image
```bash
docker build . -t "anpr-api:latest"
```

## run compose
```bash
docker-compose -f docker-compose.yaml up -d
```

## Usage
1. Run the FastAPI server by executing:
    
    uvicorn main:app --reload
    
2. Access the API at http://127.0.0.1:8000/docs
3. Upload image and get the result

## Endpoints
- /recognition/recognize: POST endpoint to upload an image of a vehicle with a number plate. Returns the recognized number plate information.

## License
This project is licensed under the MIT License

## Acknowledgements
- YOLO: You Only Look Once

Feel free to contribute or report issues in this project!

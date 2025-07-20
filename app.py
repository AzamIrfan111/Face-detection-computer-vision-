from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
from PIL import Image

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('model/yolov8_face_mask.pt')  # Replace with your model path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if file:
            # Validate file extension
            ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
                return render_template('index.html', error="Invalid file type. Use JPG, JPEG, or PNG.")

            # Save uploaded file
            filename = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Load and process image
            image = cv2.imread(file_path)
            if image is None:
                os.remove(file_path)
                return render_template('index.html', error=f"Failed to load image from {file_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 inference
            results = model(image_rgb)
            predictions = results[0].boxes
            detections = []
            annotated_image = image_rgb.copy()

            if len(predictions) > 0:
                class_names = results[0].names
                for box in predictions:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    label = class_names[int(box.cls.item())]
                    confidence = box.conf.item()
                    detections.append({'label': label, 'confidence': confidence})
                    # Draw bounding box and label
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(annotated_image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save output image
            output_filename = f"output_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Render results
            return render_template('index.html', 
                                 output_image=output_filename,
                                 detections=detections)

    return render_template('index.html')

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
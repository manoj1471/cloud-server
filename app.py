import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

# Get the directory of the current script (cloud_server.py)
base_path = os.path.dirname(os.path.abspath(__file__))

# Define paths to the YOLOv4 Tiny config, weights, and coco.names
cfg_path = os.path.join(base_path, "yolov4-tiny.cfg")
weights_path = os.path.join(base_path, "yolov4-tiny.weights")
names_path = os.path.join(base_path, "coco.names")

# Load YOLOv4-tiny model using OpenCV DNN module
yolo_net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

def detect_vehicles(image):
    height, width, channels = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    vehicle_count = 0
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                if classes[class_id] in ["car", "motorbike", "bus", "truck"]:  # Vehicle classes in COCO
                    vehicle_count += 1

    return vehicle_count

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        app.logger.debug(f"request data: {request.data}")
        data = request.get_json()
        print(f"Received Data: {data}")
        app.logger.debug(f"Parsed JSON: {data}")


        # Check if 'image' is in the request
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Get base64 encoded image
        img_data = base64.b64decode(data['image'])

        # Convert byte data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Detect vehicles in the image
        vehicle_count = detect_vehicles(image)

        return jsonify({"vehicle_count": vehicle_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

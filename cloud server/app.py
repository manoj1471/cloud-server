from flask import Flask, request, jsonify
import cv2
import numpy as np
import io

app = Flask(__name__)

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_classes = {"car", "bus", "truck", "motorbike"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_data = request.data
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        vehicle_types = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                label = classes[class_id]
                if confidence > 0.5 and label in vehicle_classes:
                    vehicle_types.append(label)

        count = len(vehicle_types)
        density = "LOW" if count <= 2 else "MEDIUM" if count <= 5 else "HIGH"
        most_common = max(set(vehicle_types), key=vehicle_types.count) if vehicle_types else "NONE"

        return jsonify({
            "count": count,
            "density": density,
            "type": most_common
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/')
def home():
    return 'YOLOv4-Tiny Flask Server is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

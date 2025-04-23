from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
vehicle_classes = ["car", "bus", "truck", "motorbike"]

# Stream URL from ESP32
esp32_url = "http://192.168.203.244:81/stream"
cap = cv2.VideoCapture(esp32_url)

# Logic
roi_y1, roi_y2 = 300, 480
light_state = "RED"
last_switch_time = time.time()
green_duration = 10

def generate_frames():
    global light_state, last_switch_time, green_duration

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w/2)
                    y = int(cy - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        vehicle_count = 0

        if len(indexes) > 0:
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                if label in vehicle_classes:
                    x, y, w, h = boxes[i]
                    cx = x + w // 2
                    cy = y + h // 2
                    if roi_y1 < cy < roi_y2:
                        vehicle_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Update traffic logic
        if vehicle_count < 5:
            green_duration = 10
        elif vehicle_count < 10:
            green_duration = 20
        else:
            green_duration = 30

        if time.time() - last_switch_time >= green_duration:
            light_state = "GREEN" if light_state == "RED" else "RED"
            last_switch_time = time.time()

        time_remaining = int(green_duration - (time.time() - last_switch_time))
        if time_remaining < 0:
            time_remaining = 0

        # Draw UI
        light_color = (0, 255, 0) if light_state == "GREEN" else (0, 0, 255)
        cv2.rectangle(frame, (0, roi_y1), (width, roi_y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Light: {light_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
        cv2.putText(frame, f"Time left: {time_remaining}s", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles in ROI: {vehicle_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

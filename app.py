from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

esp32_url = 'http://192.168.203.244:81/stream'  # Your ESP32 stream URL
cap = cv2.VideoCapture(esp32_url)

# Traffic logic
line_y = 400
already_counted = []
total_count = 0
last_switch_time = time.time()
light_state = "RED"
green_duration = 10

def generate_frames():
    global already_counted, total_count, last_switch_time, light_state, green_duration

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    x, y = int(center_x - w/2), int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        vehicles_in_roi = 0

        if len(indexes) > 0:
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                if label in vehicle_classes:
                    x, y, w, h = boxes[i]
                    cx, cy = x + w // 2, y + h // 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    if line_y - 20 < cy < line_y + 20:
                        if all(abs(cx - px) > 50 or abs(cy - py) > 50 for (px, py) in already_counted):
                            total_count += 1
                            already_counted.append((cx, cy))
                    if cy > line_y:
                        vehicles_in_roi += 1

        # Traffic light logic
        if vehicles_in_roi < 5:
            green_duration = 10
        elif vehicles_in_roi < 10:
            green_duration = 20
        else:
            green_duration = 30

        if time.time() - last_switch_time >= green_duration:
            light_state = "GREEN" if light_state == "RED" else "RED"
            last_switch_time = time.time()

        # Drawing
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Traffic Light: {light_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if light_state == "GREEN" else (0, 0, 255), 2)
        cv2.putText(frame, f"Total Vehicles: {total_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

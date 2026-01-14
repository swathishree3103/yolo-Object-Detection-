import asyncio
import json
import websockets
import cv2
import base64
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Color palette for nearest color naming
PALETTE = {
    "red": (220, 20, 60),
    "orange": (255, 128, 0),
    "yellow": (255, 215, 0),
    "green": (34, 139, 34),
    "cyan": (0, 180, 180),
    "blue": (30, 144, 255),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "brown": (150, 75, 0),
    "gray": (128, 128, 128),
    "black": (20, 20, 20),
    "white": (245, 245, 245)
}

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def nearest_color_name(rgb):
    best_name = None
    best_dist = None
    for name, pal in PALETTE.items():
        dr = rgb[0] - pal[0]
        dg = rgb[1] - pal[1]
        db = rgb[2] - pal[2]
        dist = dr*dr + dg*dg + db*db
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name

def dominant_color_from_frame(frame, k=3, resize=80):
    small = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
    data = small.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    counts = np.bincount(labels)
    largest_idx = int(np.argmax(counts))
    dom_bgr = centers[largest_idx]  # BGR order
    dom_rgb = [dom_bgr[2], dom_bgr[1], dom_bgr[0]]  # RGB order

    return {
        "rgb": [int(dom_rgb[0]), int(dom_rgb[1]), int(dom_rgb[2])],
        "hex": rgb_to_hex(dom_rgb),
        "name": nearest_color_name(dom_rgb),
        "ratio": float(counts[largest_idx]) / len(labels)
    }

async def send_detections(websocket):
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        results = model(frame)

        largest_det = None
        largest_area = 0

        # Pick the largest detected object
        for r in results[0].boxes:
            cls = int(r.cls[0])
            label = model.names[cls]
            conf = float(r.conf[0])
            x1, y1, x2, y2 = r.xyxy[0]

            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_det = {
                    "label": label,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                }

        # Find dominant color
        color_info = dominant_color_from_frame(frame)

        # Encode frame as JPEG + base64
        _, jpeg = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

        # Send JSON with largest object + color info + frame
        msg = {
            "objects": [largest_det] if largest_det else [],
            "color": color_info,
            "img_size": [frame.shape[1], frame.shape[0]],
            "frame": frame_b64
        }
        await websocket.send(json.dumps(msg))

        # Optional: show YOLO annotated feed locally
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        await asyncio.sleep(0.01)

async def main():
    print("✅ Starting YOLO WebSocket Server on ws://localhost:8765")
    async with websockets.serve(send_detections, "localhost", 8765, max_size=2**23):  # ~8MB frame limit
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
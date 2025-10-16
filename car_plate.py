import cv2
import numpy as np
from ultralytics import YOLOv10
from paddleocr import PaddleOCR

# Setting
VIDEO_PATH = "car_plate.mp4"
YOLO_MODEL_PATH = "best.pt"  # YOLOv10 path for learning file

# Class name (License plate)
CLASS_NAMES = ["License"]

# Load model
model = YOLOv10(YOLO_MODEL_PATH)

# PaddleOCR initiation
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_paddle_result(result):
    """
    PaddleOCR predict result : Output from dict
    Return texts in case over 0.5 in trust
    """
    texts = []
    rec_texts = result.get('rec_texts', [])
    rec_scores = result.get('rec_scores', [])
    for text, score in zip(rec_texts, rec_scores):
        if score > 0.8:
            texts.append(text)
    return " ".join(texts)

def process_frame(frame, frame_count):
    results = model.predict(frame, conf=0.45)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_idx = int(box.cls[0])
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"

            # Extract ROI of the plate
            roi = frame[y1:y2, x1:x2]

            # Run OCR
            ocr_result = ocr.predict(roi)[0]  # PaddleOCR return the list, the first is dict result
            text = extract_text_from_paddle_result(ocr_result)

            # Draw in the screen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{text} ({conf:.2f})" if text else f"{class_name} ({conf:.2f})"

            # Background of texts
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            print(f"[Frame {frame_count}] {class_name} box ({x1},{y1})~({x2},{y2}), conf={conf:.2f}, text='{text}'")

def run_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fail or End")
            break

        frame_count += 1
        process_frame(frame, frame_count)

        cv2.imshow("License Plate Detection + OCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # To end, press q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video()

import cv2
import numpy as np
from collections import Counter

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def dominant_color(image, k=4):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = Counter(labels.flatten())
    dominant = centers[counts.most_common(1)[0][0]]
    return tuple(map(int, dominant))

lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    box_size = 100
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    roi = frame[y1:y2, x1:x2]
    color = dominant_color(roi, k=4)
    cv2.rectangle(frame, (w - 120, 20), (w - 20, 120), color, -1)
    cv2.putText(frame, f"RGB: {color}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = mask1 + mask2
    mask = cv2.medianBlur(mask, 5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:  
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            # print(f"Tracked object center: ({x + w_box//2}, {y + h_box//2})")

    cv2.imshow("Color Detection + Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

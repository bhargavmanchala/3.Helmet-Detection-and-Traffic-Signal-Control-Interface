import cv2
from ultralytics import YOLO

model1 = YOLO('best1.pt')
model2 = YOLO('best2.pt')

def b_box_cord(values, dh, dw):
    lx = int((values[0] - values[2] / 2) * dw)
    ly = int((values[1] - values[3] / 2) * dh)
    rx = int((values[0] + values[2] / 2) * dw)
    ry = int((values[1] + values[3] / 2) * dh)
    lx = max(0, lx)
    rx = min(dw - 1, rx)
    ly = max(0, ly)
    ry = min(dh - 1, ry)
    return lx, ly, rx, ry

def predict_without_helmet(image):
    ans = True
    result2 = model2.predict(image, show_boxes=True)
    cls_m2 = result2[0].boxes.cls.tolist()
    bounding_boxes = result2[0].boxes.xywhn.tolist()
    if cls_m2.count(1) == 0:
        return True, image
    else:
        for cls_, b_box in zip(cls_m2, bounding_boxes):
            if cls_ == 1:
                ans = False
                dh, dw, _ = image.shape
                lx, ly, rx, ry = b_box_cord(b_box, dh, dw)
                cv2.rectangle(image, (lx, ly), (rx, ry), (255, 0, 0), 3)
    return ans, image

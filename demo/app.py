import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# Đường dẫn tới 2 model đã huấn luyện
MODEL1_PATH = r'd:/garbage-yolov8/gb_model/yolov8n_custom/weights/best.pt'
MODEL2_PATH = r'd:/garbage-yolov8/taco_model/yolov8s_garbage/weights/best.pt'

model1 = YOLO(MODEL1_PATH)
model2 = YOLO(MODEL2_PATH)

class_names1 = model1.names
class_names2 = model2.names

# Hàm nhận diện với cả 2 model, trả về kết quả tốt nhất

def detect_with_both_models(image):
    results1 = model1(image)[0]
    results2 = model2(image)[0]
    # Gộp box từ cả hai model
    all_boxes = list(results1.boxes) + list(results2.boxes)
    all_class_names = list(class_names1.values()) + list(class_names2.values())
    for box in all_boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = f"{all_class_names[cls_id]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

# Giao diện tkinter
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Demo nhận diện rác bằng 2 model YOLOv8')
        self.panel = tk.Label(root)
        self.panel.pack()
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text='Nhận diện Webcam', command=self.open_webcam).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text='Nhận diện Ảnh', command=self.open_image).pack(side=tk.LEFT, padx=10)
        self.stop = False

    def open_webcam(self):
        self.stop = False
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not self.stop:
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_with_both_models(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            self.root.update()
        cap.release()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg;*.jpeg;*.png')])
        if not file_path:
            return
        image = cv2.imread(file_path)
        image = detect_with_both_models(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

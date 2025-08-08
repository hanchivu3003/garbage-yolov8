import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import random


# Load mô hình YOLOv8 đã huấn luyện
model = YOLO('D:\\garbage-yolov8\\model1\yolov8n_custom\\weights\\best.pt')
class_names = model.names  # Danh sách nhãn
import random

# Tạo màu ngẫu nhiên cho từng class nhưng cố định
colors = {}
for i in range(len(model.names)):
    colors[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_label(img, text, pos, bg_color):
    """Vẽ label có nền màu"""
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Trash Detection System")
root.geometry("1200x700")  # Tăng kích thước cửa sổ

# Biến điều khiển camera
camera_active = [False]

# ========== FRAME CHÍNH ==========
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# ========== FRAME TRÁI: Nút chức năng ==========
left_frame = tk.Frame(main_frame, width=150)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

btn_image = tk.Button(left_frame, text="📷 Gửi Ảnh", command=lambda: open_image(), width=15, height=2)
btn_image.pack(pady=10)

btn_camera = tk.Button(left_frame, text="🎥 Mở Camera", command=lambda: open_camera(), width=15, height=2)
btn_camera.pack(pady=10)

btn_stop = tk.Button(left_frame, text="⛔ Tắt Camera", command=lambda: stop_camera(), width=15, height=2)
btn_stop.pack(pady=10)

# ========== FRAME GIỮA: Hiển thị ảnh/camera ==========
center_frame = tk.Frame(main_frame, bg="black")
center_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=10)

panel = tk.Label(center_frame, bg="black")
panel.pack(expand=True, fill=tk.BOTH)

# ========== FRAME PHẢI: Hiển thị kết quả ==========
right_frame = tk.Frame(main_frame, width=200)
right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

result_label = tk.Label(right_frame, text="🗑️ Các loại rác đã nhận diện:", font=("Arial", 12, "bold"))
result_label.pack(pady=(0, 5))

result_box = tk.Listbox(right_frame, width=25, height=30, font=("Arial", 11))
result_box.pack(fill=tk.BOTH, expand=True)

# ======== HÀM CHÍNH NHẬN DIỆN ==========
def detect_and_display(image):
    results = model(image)[0]

    # Clear khung kết quả
    result_box.delete(0, tk.END)
    detected_classes = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label_text = f"{class_names[cls_id]} {conf:.2f}"

        detected_classes.append(f"{class_names[cls_id]} ({conf:.2f})")

        # Vẽ khung màu riêng cho từng class
        color = colors[cls_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label_text, (x1, y1), color)

    # Hiển thị nhãn duy nhất (không lặp)
    for cls in sorted(set(detected_classes)):
        result_box.insert(tk.END, cls)

    return image


# ======== GỬI ẢNH ========
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    image = cv2.imread(file_path)
    image = detect_and_display(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    panel.config(image=imgtk)
    panel.image = imgtk

# ======== MỞ CAMERA ========
def open_camera():
    def camera_loop():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_and_display(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            panel.config(image=imgtk)
            panel.image = imgtk
            if not camera_active[0]:
                break
        cap.release()

    camera_active[0] = True
    threading.Thread(target=camera_loop, daemon=True).start()

# ======== TẮT CAMERA ========
def stop_camera():
    camera_active[0] = False

# ...existing code...

btn_video = tk.Button(left_frame, text="📹 Gửi Video", command=lambda: open_video(), width=15, height=2)
btn_video.pack(pady=10)

# ...existing code...

def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not file_path:
        return

    def video_loop():
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_and_display(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            panel.config(image=imgtk)
            panel.image = imgtk
            # Dừng nếu người dùng bấm tắt camera (dùng chung biến)
            if not camera_active[0]:
                break
            # Thêm delay để video không chạy quá nhanh
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()

    camera_active[0] = True
    threading.Thread(target=video_loop, daemon=True).start()

# ...existing code...

# Chạy GUI
root.mainloop()

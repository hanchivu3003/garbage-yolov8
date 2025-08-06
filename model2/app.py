import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# Load mô hình YOLOv8 đã huấn luyện
model = YOLO('D:\\taco_format_yolov8\\runs\\yolov8s_garbage\\weights\\best.pt')
class_names = model.names  # Danh sách nhãn

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

    # Clear khung kết quả trước mỗi lần hiển thị
    result_box.delete(0, tk.END)

    detected_classes = []  # Giữ toàn bộ nhãn (bao gồm trùng lặp)

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = f"{class_names[cls_id]} {conf:.2f}"
        detected_classes.append(f"{class_names[cls_id]} ({conf:.2f})")

        # Vẽ box và nhãn lên ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị tất cả nhãn (bao gồm trùng)
    for cls in detected_classes:
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

# Chạy GUI
root.mainloop()

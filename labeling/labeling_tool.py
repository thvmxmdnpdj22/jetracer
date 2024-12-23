import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class ImageLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling Tool")
        self.canvas = tk.Canvas(root, width=800, height=600, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        # 초기화 변수
        self.image = None
        self.image_list = []
        self.current_image_index = 0
        self.labels = {}  # {image_path: [(x1, y1, x2, y2, label), ...]}
        self.bbox_start_x = None
        self.bbox_start_y = None
        self.bbox_rect_id = None

        # 버튼과 입력 필드
        self.label_entry = tk.Entry(root, width=20)
        self.label_entry.pack(side="left", padx=10)

        self.next_button = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_button.pack(side="right", padx=10)

        self.save_button = tk.Button(root, text="Save Labels", command=self.save_labels)
        self.save_button.pack(side="left", padx=10)

        self.load_images()

        # 마우스 이벤트 연결
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def load_images(self):
        folder = "C:\Users\301\Desktop\base\jetracer\images3"
        if os.path.exists(folder):
            self.image_list = sorted(
                [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            )
            self.current_image_index = 0
            if self.image_list:
                self.display_image()
            else:
                messagebox.showerror("Error", "No images found in the folder!")
        else:
            messagebox.showerror("Error", f"Folder not found: {folder}")

    def display_image(self):
        if self.image_list:
            image_path = self.image_list[self.current_image_index]
            image = Image.open(image_path)
            image = image.resize((800, 600), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=self.image, anchor="nw")
            self.root.title(f"Labeling: {os.path.basename(image_path)}")

    def on_mouse_press(self, event):
        # 마우스 클릭 위치 저장
        self.bbox_start_x = event.x
        self.bbox_start_y = event.y
        self.bbox_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def on_mouse_drag(self, event):
        # 드래그 중 사각형 갱신
        self.canvas.coords(self.bbox_rect_id, self.bbox_start_x, self.bbox_start_y, event.x, event.y)

    def on_mouse_release(self, event):
        # 드래그 종료 시 사각형 저장
        end_x, end_y = event.x, event.y
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showwarning("Warning", "Please enter a label before adding a bounding box!")
            return

        image_path = self.image_list[self.current_image_index]
        if image_path not in self.labels:
            self.labels[image_path] = []
        self.labels[image_path].append((self.bbox_start_x, self.bbox_start_y, end_x, end_y, label))
        print(f"Bounding Box: {self.bbox_start_x, self.bbox_start_y, end_x, end_y, label}")
        self.label_entry.delete(0, tk.END)

    def next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_list):
            messagebox.showinfo("Info", "All images have been labeled!")
            self.save_labels()
            return
        self.canvas.delete("all")  # 현재 캔버스 초기화
        self.display_image()

    def save_labels(self):
        output_folder = "C:\Users\301\Desktop\base\jetracer\labeling_images3"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = os.path.join(output_folder, "labels.txt")
        try:
            with open(output_path, "w") as f:
                for image_path, bboxes in self.labels.items():
                    for bbox in bboxes:
                        x1, y1, x2, y2, label = bbox
                        f.write(f"{os.path.basename(image_path)} {x1} {y1} {x2} {y2} {label}\n")
            messagebox.showinfo("Info", f"Labels saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingTool(root)
    root.mainloop()

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import random
import time


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Face Recognition")
        self.root.geometry("1100x800")

        # Initialize models
        self.embedder = FaceNet()
        self.detector = MTCNN()
        self.classifier = None
        self.label_encoder = None

        # Minimum face size
        self.min_face_size = 50

        # Recognition variables
        self.scanning = False
        self.scan_start_time = 0
        self.scan_samples = []
        self.scan_results = []

        # Create UI
        self.create_widgets()

        # Video capture
        self.cap = None
        self.is_recognition_running = False
        self.current_frame = None

    def create_widgets(self):
        # Configure main window style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), background='#f0f0f0')

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left panel - controls
        control_frame = ttk.Frame(main_frame, width=300, style='TFrame')
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right panel - display
        self.display_frame = ttk.Frame(main_frame, style='TFrame')
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        ttk.Label(control_frame, text="Face Recognition System", style='Header.TLabel').pack(pady=15)

        # Training section
        ttk.Label(control_frame, text="Training", font=('Helvetica', 12, 'bold')).pack(pady=(20, 5), anchor='w')
        ttk.Button(control_frame, text="1. Add Training Data", command=self.add_training_data,
                   style='TButton').pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="2. Train Model", command=self.train_model,
                   style='TButton').pack(fill=tk.X, pady=5)

        # Recognition section
        ttk.Label(control_frame, text="Recognition", font=('Helvetica', 12, 'bold')).pack(pady=(20, 5), anchor='w')
        ttk.Button(control_frame, text="Start 10-Second Scan", command=self.start_scan,
                   style='TButton').pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Recognize from Photo", command=self.recognize_from_photo,
                   style='TButton').pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Stop Recognition", command=self.stop_recognition,
                   style='TButton').pack(fill=tk.X, pady=5)

        # Settings section
        ttk.Label(control_frame, text="Settings", font=('Helvetica', 12, 'bold')).pack(pady=(20, 5), anchor='w')

        # Confidence threshold
        ttk.Label(control_frame, text="Confidence Threshold:").pack(anchor='w')
        self.confidence_threshold = tk.DoubleVar(value=0.8)  # Set to 80%
        ttk.Scale(control_frame, from_=0.5, to=1.0, variable=self.confidence_threshold,
                  orient=tk.HORIZONTAL, length=200).pack(fill=tk.X, pady=5)

        # Minimum face size
        ttk.Label(control_frame, text="Min Face Size:").pack(anchor='w')
        self.min_face_size_var = tk.IntVar(value=self.min_face_size)
        ttk.Scale(control_frame, from_=20, to=100, variable=self.min_face_size_var,
                  orient=tk.HORIZONTAL, length=200, command=self.update_min_face_size).pack(fill=tk.X, pady=5)

        # Scan progress
        self.scan_progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.scan_progress.pack(fill=tk.X, pady=10)
        self.scan_progress_label = ttk.Label(control_frame, text="Ready to scan", anchor='center')
        self.scan_progress_label.pack(fill=tk.X)

        # Exit button
        ttk.Button(control_frame, text="Exit", command=self.on_close,
                   style='TButton').pack(fill=tk.X, pady=(30, 5))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor='center').pack(fill=tk.X, pady=10, side=tk.BOTTOM)

        # Image label for display
        self.image_label = ttk.Label(self.display_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def update_min_face_size(self, event=None):
        """Update the min_face_size when slider changes"""
        self.min_face_size = self.min_face_size_var.get()

    def start_scan(self):
        if not os.path.exists('face_recognition_model.pkl'):
            messagebox.showerror("Error", "Model not found! Train first.")
            return

        if self.scanning:
            return

        self.scanning = True
        self.scan_samples = []
        self.scan_results = []
        self.scan_start_time = time.time()
        self.scan_progress['value'] = 0
        self.scan_progress_label.config(text="Scanning... 0%")

        if not self.is_recognition_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                self.scanning = False
                return
            self.is_recognition_running = True

        self.status_var.set("10-second scan in progress")
        self.update_scan()

    def update_scan(self):
        if not self.scanning:
            return

        elapsed = time.time() - self.scan_start_time
        progress = min(100, (elapsed / 10) * 100)
        self.scan_progress['value'] = progress
        self.scan_progress_label.config(text=f"Scanning... {int(progress)}%")

        if elapsed >= 10:
            self.complete_scan()
            return

        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                detections = self.detector.detect_faces(rgb_frame)
                for detection in detections:
                    x, y, w, h = detection['box']
                    if w < self.min_face_size or h < self.min_face_size:
                        continue

                    x, y = abs(x), abs(y)
                    face = rgb_frame[y:y + h, x:x + w]

                    try:
                        face = cv2.resize(face, (160, 160))
                        self.scan_samples.append(face)

                        # Show scanning frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, "Scanning...",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    except:
                        pass

            except Exception as e:
                print(f"Scan error: {e}")

            self.current_frame = frame
            self.update_display()

        self.root.after(100, self.update_scan)

    def complete_scan(self):
        self.scanning = False
        self.scan_progress['value'] = 100
        self.scan_progress_label.config(text="Scan complete")

        if not self.scan_samples:
            messagebox.showerror("Error", "No faces detected during scan!")
            self.status_var.set("Scan failed - no faces")
            return

        try:
            # Process all collected samples
            samples = np.array(self.scan_samples)
            embeddings = self.embedder.embeddings(samples)

            for embedding in embeddings:
                prediction = self.classifier.predict_proba([embedding])[0]
                best_class_idx = np.argmax(prediction)
                confidence = prediction[best_class_idx]

                if confidence >= self.confidence_threshold.get():
                    person_name = self.label_encoder.inverse_transform([best_class_idx])[0]
                    self.scan_results.append((person_name, confidence))

            if not self.scan_results:
                messagebox.showerror("Recognition Failed",
                                     "Face not recognized with sufficient confidence (need >80%)")
                self.status_var.set("Scan failed - low confidence")
            else:
                # Get the most common result
                names = [r[0] for r in self.scan_results]
                best_name = max(set(names), key=names.count)
                avg_confidence = sum(r[1] for r in self.scan_results if r[0] == best_name) / names.count(best_name)

                if avg_confidence >= 0.8:
                    messagebox.showinfo("Recognition Successful",
                                        f"Recognized: {best_name}\nAverage Confidence: {avg_confidence:.2%}")
                    self.status_var.set(f"Recognized: {best_name}")
                else:
                    messagebox.showerror("Recognition Failed",
                                         "Face not recognized with sufficient confidence (need >80%)")
                    self.status_var.set("Scan failed - low confidence")

        except Exception as e:
            messagebox.showerror("Error", f"Scan processing failed: {str(e)}")
            self.status_var.set("Scan failed - error")

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            detections = self.detector.detect_faces(rgb_frame)

            for detection in detections:
                x, y, w, h = detection['box']
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                x, y = abs(x), abs(y)
                face = rgb_frame[y:y + h, x:x + w]

                try:
                    face = cv2.resize(face, (160, 160))
                    embedding = self.embedder.embeddings([face])[0]
                    prediction = self.classifier.predict_proba([embedding])[0]
                    best_class_idx = np.argmax(prediction)
                    confidence = prediction[best_class_idx]

                    if confidence >= self.confidence_threshold.get():
                        person_name = self.label_encoder.inverse_transform([best_class_idx])[0]
                        color = (0, 255, 0)  # Green
                    else:
                        person_name = "Unknown"
                        color = (0, 0, 255)  # Red

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{person_name} ({confidence:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception as e:
                    print(f"Recognition error: {e}")

        except Exception as e:
            print(f"Detection error: {e}")

        return frame

    def simple_augment(self, image):
        """Custom data augmentation without imgaug"""
        augmented = []

        # Original image
        augmented.append(image)

        # Horizontal flip
        augmented.append(cv2.flip(image, 1))

        # Small rotations (-15 to 15 degrees)
        rows, cols = image.shape[:2]
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented.append(rotated)

        # Slight blur
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        augmented.append(blurred)

        # Brightness adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.7, 1.3)
        hsv = np.clip(hsv, 0, 255)
        bright = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        augmented.append(bright)

        return augmented

    def train_model(self):
        if not os.path.exists("training_data"):
            messagebox.showerror("Error", "No training_data folder found!")
            return

        def training_thread():
            try:
                self.status_var.set("Training in progress...")
                train_faces = []
                train_labels = []

                for person_name in os.listdir("training_data"):
                    person_dir = os.path.join("training_data", person_name)

                    for img_name in os.listdir(person_dir):
                        img_path = os.path.join(person_dir, img_name)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        try:
                            results = self.detector.detect_faces(img)
                            if not results:
                                continue

                            x1, y1, width, height = results[0]['box']
                            x1, y1 = abs(x1), abs(y1)
                            x2, y2 = x1 + width, y1 + height
                            face = img[y1:y2, x1:x2]
                            face = cv2.resize(face, (160, 160))

                            # Apply custom augmentation
                            augmented_faces = self.simple_augment(face)
                            for aug_face in augmented_faces:
                                train_faces.append(aug_face)
                                train_labels.append(person_name)
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")
                            continue

                if not train_faces:
                    messagebox.showerror("Error", "No faces found in training data!")
                    self.status_var.set("Training failed")
                    return

                # Convert to numpy arrays
                train_faces = np.array(train_faces)
                embeddings = self.embedder.embeddings(train_faces)

                # Train classifier
                self.label_encoder = LabelEncoder()
                train_labels_encoded = self.label_encoder.fit_transform(train_labels)

                self.classifier = SVC(kernel='linear', probability=True)
                self.classifier.fit(embeddings, train_labels_encoded)

                # Save model
                with open('face_recognition_model.pkl', 'wb') as f:
                    pickle.dump((self.classifier, self.label_encoder), f)

                messagebox.showinfo("Success", f"Model trained with {len(train_faces)} samples!")
                self.status_var.set("Model trained")
            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {str(e)}")
                self.status_var.set("Training failed")

        threading.Thread(target=training_thread, daemon=True).start()

    def start_recognition(self):
        if self.is_recognition_running:
            return

        try:
            with open('face_recognition_model.pkl', 'rb') as f:
                self.classifier, self.label_encoder = pickle.load(f)
        except:
            messagebox.showerror("Error", "Model not found! Train first.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        self.is_recognition_running = True
        self.status_var.set("Webcam recognition running")
        self.update_recognition()

    def recognize_from_photo(self):
        if not os.path.exists('face_recognition_model.pkl'):
            messagebox.showerror("Error", "Model not found! Train first.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        try:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Could not read the image!")
                return

            processed_img = self.recognize_faces(img.copy())
            self.display_photo_result(processed_img)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def display_photo_result(self, frame):
        result_window = tk.Toplevel(self.root)
        result_window.title("Recognition Result")
        result_window.geometry("800x600")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        label = ttk.Label(result_window, image=img)
        label.image = img
        label.pack(fill=tk.BOTH, expand=True)

        ttk.Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)

    def update_recognition(self):
        if not self.is_recognition_running:
            return

        ret, frame = self.cap.read()
        if ret:
            processed_frame = self.recognize_faces(frame)
            self.current_frame = processed_frame
            self.update_display()

        self.root.after(10, self.update_recognition)

    def update_display(self):
        if self.current_frame is not None:
            img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.image_label.configure(image=img)
            self.image_label.image = img

    def stop_recognition(self):
        self.scanning = False
        self.is_recognition_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_var.set("Recognition stopped")
        self.scan_progress['value'] = 0
        self.scan_progress_label.config(text="Ready to scan")

    def add_training_data(self):
        folder_path = filedialog.askdirectory(title="Select Person Folder")
        if not folder_path:
            return

        person_name = os.path.basename(folder_path)
        dest_folder = os.path.join("training_data", person_name)

        if not os.path.exists("training_data"):
            os.makedirs("training_data")

        if os.path.exists(dest_folder):
            if not messagebox.askyesno("Confirm", f"Person '{person_name}' already exists. Add to existing data?"):
                return

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        added_count = 0
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                src = os.path.join(folder_path, img_file)
                dst = os.path.join(dest_folder, img_file)
                try:
                    img = cv2.imread(src)
                    if img is not None:
                        cv2.imwrite(dst, img)
                        added_count += 1
                except:
                    continue

        messagebox.showinfo("Success", f"Added {added_count} images of {person_name} to training data")

    def on_close(self):
        self.stop_recognition()
        self.root.destroy()


if __name__ == "__main__":
    # Check for required folders
    if not os.path.exists("training_data"):
        os.makedirs("training_data")

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
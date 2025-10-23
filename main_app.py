import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

# Import our custom modules
from face_detector import FaceDetector
from adversarial_modifier import AdversarialFaceModifier
from privacy_evaluator import PrivacyEvaluator

class PrivacyFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Privacy-Preserving Face Detection System")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_modifier = AdversarialFaceModifier()
        self.privacy_evaluator = PrivacyEvaluator()
        
        # Camera variables
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.processed_frame = None
        
        # Processing settings
        self.modification_method = tk.StringVar(value="perturbation")
        self.noise_factor = tk.DoubleVar(value=0.1)
        self.blur_kernel = tk.IntVar(value=7)
        self.pixel_size = tk.IntVar(value=8)
        self.show_detection = tk.BooleanVar(value=True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Video display area
        self.setup_video_display(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel with buttons and settings"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera", padding="5")
        camera_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_camera_btn = ttk.Button(camera_frame, text="Start Camera", 
                                         command=self.start_camera)
        self.start_camera_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_camera_btn = ttk.Button(camera_frame, text="Stop Camera", 
                                        command=self.stop_camera, state="disabled")
        self.stop_camera_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.load_image_btn = ttk.Button(camera_frame, text="Load Image", 
                                       command=self.load_image)
        self.load_image_btn.grid(row=0, column=2)
        
        # Processing method selection
        method_frame = ttk.LabelFrame(control_frame, text="Privacy Method", padding="5")
        method_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        methods = [("Perturbation", "perturbation"), ("Blur", "blur"), ("Pixelate", "pixelate")]
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(method_frame, text=text, variable=self.modification_method, 
                           value=value, command=self.on_method_change).grid(row=0, column=i, padx=5)
        
        # Method-specific parameters
        params_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Perturbation parameters
        ttk.Label(params_frame, text="Noise Factor:").grid(row=0, column=0, sticky=tk.W)
        noise_scale = ttk.Scale(params_frame, from_=0.01, to=0.5, variable=self.noise_factor, 
                               orient=tk.HORIZONTAL, length=150)
        noise_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # Blur parameters
        ttk.Label(params_frame, text="Blur Kernel:").grid(row=1, column=0, sticky=tk.W)
        blur_scale = ttk.Scale(params_frame, from_=3, to=21, variable=self.blur_kernel, 
                              orient=tk.HORIZONTAL, length=150)
        blur_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # Pixelation parameters
        ttk.Label(params_frame, text="Pixel Size:").grid(row=2, column=0, sticky=tk.W)
        pixel_scale = ttk.Scale(params_frame, from_=4, to=20, variable=self.pixel_size, 
                               orient=tk.HORIZONTAL, length=150)
        pixel_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # Display options
        display_frame = ttk.LabelFrame(control_frame, text="Display", padding="5")
        display_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(display_frame, text="Show Face Detection", 
                       variable=self.show_detection).grid(row=0, column=0, sticky=tk.W)
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(action_frame, text="Save Image", 
                  command=self.save_image).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(action_frame, text="Evaluate Privacy", 
                  command=self.evaluate_privacy).grid(row=0, column=1, padx=(5, 0))
    
    def setup_video_display(self, parent):
        """Setup video display area"""
        display_frame = ttk.LabelFrame(parent, text="Video Feed", padding="10")
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas for video display
        self.video_canvas = tk.Canvas(display_frame, width=640, height=480, bg="black")
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
    
    def setup_status_bar(self, parent):
        """Setup status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
            
            self.camera_running = True
            self.start_camera_btn.config(state="disabled")
            self.stop_camera_btn.config(state="normal")
            self.status_var.set("Camera started")
            
            # Start video processing thread
            self.video_thread = threading.Thread(target=self.process_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        
        self.start_camera_btn.config(state="normal")
        self.stop_camera_btn.config(state="disabled")
        self.status_var.set("Camera stopped")
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_frame = image
                    self.process_single_image()
                    self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "Could not load image")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def process_video(self):
        """Process video frames in separate thread"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.process_single_image()
            else:
                break
    
    def process_single_image(self):
        """Process a single image with face detection and modification"""
        if self.current_frame is None:
            return
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(self.current_frame)
            
            # Create processed frame
            processed = self.current_frame.copy()
            
            if faces:
                # Extract face regions
                face_regions = self.face_detector.extract_face_regions(self.current_frame, faces)
                
                # Apply privacy modification to each face
                for i, (face_region, (x, y, w, h)) in enumerate(zip(face_regions, faces)):
                    # Get modification parameters
                    method = self.modification_method.get()
                    
                    if method == "perturbation":
                        modified_face = self.face_modifier.modify_face(
                            face_region, method, noise_factor=self.noise_factor.get()
                        )
                    elif method == "blur":
                        kernel_size = int(self.blur_kernel.get())
                        if kernel_size % 2 == 0:  # Ensure odd kernel size
                            kernel_size += 1
                        modified_face = self.face_modifier.modify_face(
                            face_region, method, kernel_size=kernel_size, sigma=2.0
                        )
                    elif method == "pixelate":
                        modified_face = self.face_modifier.modify_face(
                            face_region, method, pixel_size=int(self.pixel_size.get())
                        )
                    else:
                        modified_face = face_region
                    
                    # Replace face region in processed frame
                    padding = 10
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(processed.shape[1], x + w + padding)
                    y_end = min(processed.shape[0], y + h + padding)
                    
                    # Resize modified face to match region size
                    region_h = y_end - y_start
                    region_w = x_end - x_start
                    modified_resized = cv2.resize(modified_face, (region_w, region_h))
                    
                    processed[y_start:y_end, x_start:x_end] = modified_resized
            
            # Draw detection rectangles if enabled
            if self.show_detection.get() and faces:
                processed = self.face_detector.draw_face_rectangles(processed, faces)
            
            self.processed_frame = processed
            
            # Update display
            self.root.after(0, self.update_display)
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def update_display(self):
        """Update the video display"""
        if self.processed_frame is not None:
            # Resize frame to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_resized = cv2.resize(self.processed_frame, (canvas_width, canvas_height))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                image_pil = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image_pil)
                
                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.video_canvas.image = photo  # Keep a reference
    
    def on_method_change(self):
        """Handle modification method change"""
        self.status_var.set(f"Method changed to: {self.modification_method.get()}")
    
    def save_image(self):
        """Save current processed image"""
        if self.processed_frame is not None:
            file_path = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            )
            
            if file_path:
                try:
                    cv2.imwrite(file_path, self.processed_frame)
                    self.status_var.set(f"Saved: {os.path.basename(file_path)}")
                    messagebox.showinfo("Success", "Image saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}")
        else:
            messagebox.showwarning("Warning", "No image to save")
    
    def evaluate_privacy(self):
        """Evaluate privacy protection effectiveness"""
        if self.current_frame is not None and self.processed_frame is not None:
            try:
                evaluation = self.privacy_evaluator.evaluate_privacy_protection(
                    self.current_frame, self.processed_frame
                )
                
                # Show evaluation results in a dialog
                self.show_evaluation_dialog(evaluation)
                
            except Exception as e:
                messagebox.showerror("Error", f"Privacy evaluation failed: {e}")
        else:
            messagebox.showwarning("Warning", "No images to evaluate")
    
    def show_evaluation_dialog(self, evaluation):
        """Show privacy evaluation results in a dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Privacy Evaluation Results")
        dialog.geometry("500x400")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, width=60, height=20)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add evaluation results
        results_text = f"""Privacy Protection Evaluation Results
        
Original faces recognized: {evaluation['original_faces_recognized']}
Modified faces recognized: {evaluation['modified_faces_recognized']}
Privacy protection rate: {evaluation['privacy_protection_rate']:.2%}
Privacy protection successful: {evaluation['privacy_successful']}
Image quality (PSNR): {evaluation['image_quality_psnr']:.2f} dB
Structural similarity: {evaluation['structural_similarity']:.3f}

Interpretation:
- Higher privacy protection rate is better
- Privacy successful = True means no faces were recognized in modified image
- Higher PSNR means better image quality (>30 dB is good)
- Higher structural similarity means the image looks more like the original
"""
        
        text_widget.insert(tk.END, results_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))
    
    def __del__(self):
        """Cleanup when app is destroyed"""
        if self.cap:
            self.cap.release()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = PrivacyFaceApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        if hasattr(app, 'cap') and app.cap:
            app.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
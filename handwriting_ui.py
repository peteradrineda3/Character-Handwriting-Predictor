import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn

class HandwritingApp:
    def __init__(self, master):
        self.master = master
        master.title("Handwriting Recognizer")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #loading model
        self.model = self.load_model()
        self.class_names = [str(i) for i in range(10)] + \
                          [chr(i) for i in range(65, 91)] + \
                          [chr(i) for i in range(97, 123)]
        
        #making UI elements
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.label = tk.Label(master, text="Draw a character", font=('Helvetica', 18))
        self.label.pack()
        
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(pady=10)
        
        self.clear_btn = tk.Button(self.btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.predict_btn = tk.Button(self.btn_frame, text="Predict", command=self.predict)
        self.predict_btn.pack(side=tk.RIGHT, padx=10)
        
        #drawing setup
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        
        #event bindings
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.reset_position)
    
    def load_model(self):
        #making sure model matches the training architecture
        class CNNPredict(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64*7*7, 128), nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 62)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = CNNPredict().to(self.device)
        model.load_state_dict(torch.load('emnist_model.pth', map_location=self.device))
        model.eval()
        return model
    
    def draw_line(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                   width=15, fill='black', capstyle=tk.ROUND)
            self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=15)
        self.last_x = x
        self.last_y = y
    
    def reset_position(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a character")
    
    def preprocess_image(self):
        img = self.image.resize((28, 28)).convert('L')
        img = ImageOps.invert(img)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        return img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def predict(self):
        img_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(img_tensor)
        
        probabilities = nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
        self.label.config(text=f"Prediction: {self.class_names[predicted.item()]} ({confidence.item():.2%})")

if __name__ == '__main__':
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
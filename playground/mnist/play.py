import tkinter as tk
import numpy as np
from tensorflow import keras
from omni import Model
from pathlib import Path

GRID_SIZE = 28
CELL_SIZE = 20
CANVAS_SIZE = GRID_SIZE * CELL_SIZE


class MNISTDrawGUI:
    def __init__(self, root, tf_model, omni_model):
        self.omni_model = omni_model
        self.tf_model = tf_model
        
        self.root = root
        self.root.title("MNIST Drawer")

        # ===== MAIN LAYOUT =====
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        # ===== LEFT: DRAWING AREA =====
        self.canvas = tk.Canvas(
            self.main_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white"
        )
        self.canvas.grid(row=0, column=0)

        # ===== RIGHT: CONTROL PANEL =====
        self.right_panel = tk.Frame(
            self.main_frame,
            width=200,
            bg="white"
        )
        self.right_panel.grid(row=0, column=1, padx=20, sticky="n")

        # Labels for predictions
        self.pred_label_title = tk.Label(
            self.right_panel,
            text="Model Prediction",
            font=("Arial", 12, "bold"),
            bg="white"
        )
        self.pred_label_title.pack(anchor="w", pady=(10, 5))

        self.omni_pred_label_1 = tk.Label(
            self.right_panel,
            text="omni Digit: -",
            font=("Arial", 11),
            bg="white"
        )
        self.omni_pred_label_1.pack(anchor="w")

        self.omni_pred_label_2 = tk.Label(
            self.right_panel,
            text="omni Confidence: -",
            font=("Arial", 11),
            bg="white"
        )
        self.omni_pred_label_2.pack(anchor="w", pady=(0, 20))
        
        self.tf_pred_label_1 = tk.Label(
            self.right_panel,
            text="tf Digit: -",
            font=("Arial", 11),
            bg="white"
        )
        self.tf_pred_label_1.pack(anchor="w")

        self.tf_pred_label_2 = tk.Label(
            self.right_panel,
            text="tf Confidence: -",
            font=("Arial", 11),
            bg="white"
        )
        self.tf_pred_label_2.pack(anchor="w", pady=(0, 20))


        # Predict button
        self.predict_button = tk.Button(
            self.right_panel,
            text="Predict",
            width=15,
            command=self.predict
        )
        self.predict_button.pack(pady=5)

        # Clear button
        self.clear_button = tk.Button(
            self.right_panel,
            text="Clear",
            width=15,
            command=self.clear
        )
        self.clear_button.pack(pady=5)

        # ===== INTERNAL GRID =====
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.draw_grid()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

    # ===== DRAWING =====
    def draw_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0 = j * CELL_SIZE
                y0 = i * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    outline="lightgray",
                    fill="white",
                    tags=f"cell_{i}_{j}"
                )

    def draw(self, event):
        row = event.y // CELL_SIZE
        col = event.x // CELL_SIZE

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self.paint_cell(row, col, 1.0)

            # Thicker stroke MNIST-like
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = row + dr, col + dc
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        self.paint_cell(r, c, 0.5)

    def paint_cell(self, r, c, value):
        self.grid[r, c] = max(self.grid[r, c], value)
        self.canvas.itemconfig(f"cell_{r}_{c}", fill="black")


    # ===== BUTTON ACTIONS =====
    def predict(self):
        """
        Replace this with your actual model call
        """
        x = self.grid.flatten()  # (784,)
        x = x / 1.0

        omni_y = self.omni_model.predict(x)
        omni_digit = np.argmax(omni_y)

        tf_y = self.tf_model.predict(np.expand_dims(x, axis=0))
        tf_digit = np.argmax(tf_y[0])
       

        self.omni_pred_label_1.config(text=f"omni Digit: {omni_digit}")
        self.omni_pred_label_2.config(text=f"omni Confidence: {omni_y[omni_digit]:.2f}")
        
        self.tf_pred_label_1.config(text=f"tf Digit: {tf_digit}")
        self.tf_pred_label_2.config(text=f"tf Confidence: {tf_y[0][omni_digit]:.2f}")

    def clear(self):
        self.grid.fill(0.0)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.itemconfig(f"cell_{i}_{j}", fill="white")

        self.omni_pred_label_1.config(text="omni Digit: -")
        self.omni_pred_label_2.config(text="omni Confidence: -")
        self.tf_pred_label_1.config(text="tf Digit: -")
        self.tf_pred_label_2.config(text="tf Confidence: -")


if __name__ == "__main__":
    root = tk.Tk()
    
    script_dir = Path(__file__).parent.resolve()
    
    omni_model = Model.load(script_dir/"model"/"omni_mnist.model")
    tf_model = keras.models.load_model(script_dir/"model"/"tf_mnist.keras")
    app = MNISTDrawGUI(root, omni_model=omni_model, tf_model=tf_model)
    
    root.mainloop()

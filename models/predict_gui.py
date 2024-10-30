import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

def predict_from_saved_model(model_path, scaler_path, input_values, feature_names):
    # Load the model and scaler from files
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare input DataFrame and scale it
    manual_input = pd.DataFrame([input_values], columns=feature_names)
    scaled_input = scaler.transform(manual_input)
    
    # Predict using the loaded model
    prediction = model.predict(scaled_input)
    return prediction[0]

def get_prediction():
    try:
        # Get feature values from entry fields
        input_values = [float(entry.get()) for entry in entries]
        
        # Define file paths for the saved model and scaler
        model_path = "stacking_regressor_model.joblib"
        scaler_path = "scaler.joblib"
        
        # Define feature names
        feature_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        # Get prediction
        prediction = predict_from_saved_model(model_path, scaler_path, input_values, feature_names)
        
        # Display result in the result entry (clear existing text, then insert new)
        result_entry.config(state="normal")  # Make it writable
        result_entry.delete(0, tk.END)
        result_entry.insert(0, f"{prediction:.2f}")
        result_entry.config(state="readonly")  # Make it readonly for copying
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def reset_fields():
    # Clear all input fields and the result entry
    for entry in entries:
        entry.delete(0, tk.END)
    result_entry.config(state="normal")  # Make it writable
    result_entry.delete(0, tk.END)
    result_entry.config(state="readonly")  # Set it back to readonly

# Set up the main GUI window
root = tk.Tk()
root.title("AOR Prediction")
root.geometry("300x550")  # Adjusted initial size
root.configure(bg="#2e3b4e")  # Darker background color

# Frame to hold input fields
frame = ttk.Frame(root, padding=20, style="TFrame")
frame.pack(fill="both", expand=True, padx=20, pady=20)

# Header Label
header_label = ttk.Label(frame, text="Enter Feature Values", font=("Helvetica", 16, "bold"), foreground="#f0f4f7")
header_label.pack(pady=(0, 20))

# Feature input fields, arranged in rows with labels and entries side-by-side
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
entries = []

# Create a row frame for each label and entry
for label_text in labels:
    row_frame = ttk.Frame(frame)
    row_frame.pack(fill="x", pady=5)

    label = ttk.Label(row_frame, text=f"{label_text}:", font=("Helvetica", 11), foreground="#f0f4f7")
    label.pack(side="left", padx=(0, 10))

    entry = ttk.Entry(row_frame, font=("Helvetica", 11), width=20)
    entry.pack(side="right", fill="x", expand=True)
    entries.append(entry)

# Predict button
predict_button = ttk.Button(frame, text="Get Prediction", command=get_prediction, style="Accent.TButton")
predict_button.pack(pady=10)

# Result entry for displaying prediction result
result_label = ttk.Label(frame, text="Predicted AOR:", font=("Helvetica", 12), foreground="#f0f4f7", background="#2e3b4e")
result_label.pack(pady=(10, 0))

result_entry = ttk.Entry(frame, font=("Helvetica", 12), width=20, justify="center", state="readonly")
result_entry.pack(pady=5)

# Reset button
reset_button = ttk.Button(frame, text="Reset", command=reset_fields, style="Accent.TButton")
reset_button.pack(pady=10)

# Styling options for a more modern, subdued look
style = ttk.Style()
style.configure("TFrame", background="#2e3b4e")
style.configure("TLabel", background="#2e3b4e", font=("Helvetica", 11))
style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), foreground="#ffffff", background="#5b9bd5", padding=6)
style.map("Accent.TButton", background=[("active", "#005f99")])

# Run the GUI loop
root.mainloop()

import tkinter as tk
import unicodedata

def convert_text():
    # Get the input text
    input_text = entry.get()
    # Convert the text to lowercase
    converted_text = input_text.lower()
    # Normalize the text to ensure it's in normal form
    normalized_text = unicodedata.normalize('NFKC', converted_text)
    # Display the normalized text in the result entry box
    result_entry.delete(0, tk.END)  # Clear the result entry box
    result_entry.insert(0, normalized_text)  # Insert the normalized text

# Create the main window
root = tk.Tk()
root.title("Text Converter")

# Create a label for instructions
instruction_label = tk.Label(root, text="Enter text to convert:")
instruction_label.pack(pady=5)

# Create an entry box for user input
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

# Create a button to trigger the conversion
convert_button = tk.Button(root, text="Convert", command=convert_text)
convert_button.pack(pady=5)

# Create an entry box to display the result (allows copying)
result_entry = tk.Entry(root, width=30)
result_entry.pack(pady=5)

# Run the application
root.mainloop()
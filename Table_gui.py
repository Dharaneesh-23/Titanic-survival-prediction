import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import Random_forest as RF
def load_csv():
    # Read CSV file
    csv_file = "C:/Users/Dell/Desktop/project/ML Project/titanic/test.csv"
    dataframe = pd.read_csv(csv_file)
    
    # Clear previous data from the table
    for i in treeview.get_children():
        treeview.delete(i)
    
    # Populate the table with CSV data
    for index, row in dataframe.iterrows():
        treeview.insert("", "end", values=list(row))

def show_details(event):
    # Get the selected row
    selected_item = treeview.focus()
    if selected_item:
        # Get the values of the selected row
        values = treeview.item(selected_item, "values")
        
        # Create a new window for displaying row details
        details_window = tk.Toplevel()
        details_window.title("Row Details")
        str = ""
        # Display the row details
        cl = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
        for i, col in enumerate(columns):
            label = tk.Label(details_window, text=f"{col}: {values[i]}")
            label.pack()
            if(col in cl):
                if(col == "Sex"):
                    if(values[i] == "female"):
                        str = str+"0,"
                    elif(values[i] == "male"):
                        str = str+"1,"
                elif(col == "Embarked"):
                    if(values[i] == "C"):
                        str = str+"0,"
                    elif(values[i] == "Q"):
                        str = str+"1,"
                    elif(values[i] == "S"):
                        str = str+"2,"
                else:
                    str = str+values[i]+","
        #print(str)
        if(str[-1] == ","):
            n=len(str)
            str = str[:n-1]
            #print(str)
        # Add a button for prediction
        st = RF.Random_forest_analysis(str)
        label1=tk.Label(details_window, text=f"Prediction: {st}")
        label1.pack()

# Create the main window
window = tk.Tk()
window.title("CSV Viewer")

# Create a Treeview widget for displaying the table
treeview = ttk.Treeview(window)

# Add columns to the Treeview based on CSV headers (assumed as first row)
csv_file = "C:/Users/Dell/Desktop/project/ML Project/titanic/test.csv"
dataframe = pd.read_csv(csv_file, nrows=1)
columns = dataframe.columns.tolist()
treeview["columns"] = columns

# Configure column headings
treeview.heading("#0", text="Index")
treeview.column("#0", width=50)
for col in columns:
    treeview.heading(col, text=col)
    treeview.column(col, width=100)

treeview.pack(fill="both", expand=True)

# Create a button to load the CSV file
load_button = tk.Button(window, text="Load CSV", command=load_csv)
load_button.pack()

label = tk.Label(window,text = f"Accuracy: {RF.accuray()}")
label.place(relx = 0.0, rely = 1.0,anchor ='sw')
label.pack()

# Bind the show_details function to the selection event
treeview.bind("<<TreeviewSelect>>", show_details)

# Start the GUI event loop
window.mainloop()

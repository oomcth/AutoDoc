import tkinter as tk
from gui import AudioChatInterface

start_script = "./start_record.sh"  # ou .bat sur Windows
stop_script = "./end_record.sh"
root = tk.Tk()
root.geometry("1000x1000")
app = AudioChatInterface(root, start_script, stop_script)
root.mainloop()

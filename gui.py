import tkinter as tk

root = tk.Tk() # Create GUI window 

### WIDGETS ###
# NOTE: Change to themed tk widgets in the future if I'm bored
frame = tk.Frame(master=root, width=300, height=300)
frame.pack()


title = tk.Label(master=frame, text="Hopper Simulation")
title.pack()

btnMass = tk.Button(master=frame, text="Set Mass", command=inputMass)


root.mainloop() # Run event loop

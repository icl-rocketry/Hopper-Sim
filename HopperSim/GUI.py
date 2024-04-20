import tkinter as tk
from tkinter import ttk

class GUI(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Basic Hopper Simulator")
    
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (menuPage, initialConditions):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(menuPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class menuPage(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page")
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Initial Conditions",
                            command=lambda: controller.show_frame(initialConditions))
        button.pack()



class initialConditions(tk.Frame):
    pass


# Run script if exectued directly (ONLY FOR TESTING)
if __name__ == "__main__":
    app = GUI()
    app.mainloop()
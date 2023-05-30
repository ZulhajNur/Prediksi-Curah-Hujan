import time
import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

import Program as p

canvas = None  

root = tk.Tk()
root.title("Number Recognition")
root.wm_attributes('-transparentcolor', '#ab23ff')

mainFrame = tk.Frame(root, width=800, height=326)
mainFrame.grid(), mainFrame.grid_propagate(0)


frame1    = tk.LabelFrame(mainFrame, width=265, height=320)
frame1.grid(row = 0, column=0, padx=3, pady=3)

labelImg  = tk.Label(frame1, image=tk.PhotoImage(width=256, height=256))
labelImg.grid(row=0)
labelRes  = tk.Label(frame1, text='Click button below')
labelRes.grid(row=1)
buttonIm  = tk.Button(frame1, text='Randomize Number', command=p.simulation)
buttonIm.grid(row=2, pady=5)


frame2      = tk.LabelFrame(mainFrame, width=90, height=320)
frame2.grid(row=0, column=1, padx=3, pady=3)

frame2_1    = tk.LabelFrame(frame2, width=90, height=320)
frame2_1.grid(row=0)

labelImgOut1 = tk.Label(frame2_1, image=tk.PhotoImage(width=25, height=256))
labelImgOut1.grid(row=0, column=0)

for i in range(0, 10):
    tk.Label(labelImgOut1, text='{}'.format(i)).grid(row=i, column=0, pady=2)
labelImgOut2 = tk.Label(frame2_1, image=tk.PhotoImage(width=25, height=256))
labelImgOut2.grid(row=0, column=1)
labelRes2   = tk.Label(frame2, text='What is this?')
labelRes2.grid(row=1)
button2     = tk.Button(frame2, text='Reset')
button2.grid(row=2, pady=3)

frame4    = tk.LabelFrame(mainFrame, width=254, height=321)
frame4.grid(row=0, column=2, padx=3, pady=3), frame4.grid_propagate(0)

label4_1  = tk.Label(frame4, text='')
label4_1.grid(row=0, sticky='W')
label4_2  = tk.Label(frame4, text='')
label4_2.grid(row=1, sticky='W')
label4_3  = tk.Label(frame4, text='')
label4_3.grid(row=2, sticky='W')
label4_4  = tk.Label(frame4, text='Simulation-{}'.format(p.num))
label4_4.grid(row=3, sticky='W')
label4_5  = tk.Label(frame4, text='Recognition correct = {}/{}'.format(p.cor, p.num))
label4_5.grid(row=4, sticky='W')
label4_6  = tk.Label(frame4, text='Cost = {}'.format('0'))
label4_6.grid(row=5, sticky='W')
cnvFrame = tk.Frame(frame4)
cnvFrame.grid(row=6)

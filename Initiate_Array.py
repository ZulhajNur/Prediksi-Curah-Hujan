import time
import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import openpyxl as op

layerInput  = np.zeros((30))
layer1      = np.zeros((16))
layer2      = np.zeros((16))
layerOutput = np.zeros((30))

# Weight
w0 = np.zeros((31, 16))    # Input - Hidden layer 1
w1 = np.zeros((16, 16))    # Hidden Layer 1 - Hidden Layer 2
w2 = np.zeros((16, 31))    # Hidden Layer 2 - Output

# Bias
b0 = np.zeros((16))        # Input - Hidden layer 1
b1 = np.zeros((16))        # Hidden Layer 1 - Hidden Layer 2
b2 = np.zeros((31))        # Hidden Layer 2 - Output

# Node dan Keluaran Activation Function
n0 = np.zeros((31))        # Input
nx = np.zeros((31))        # Input
n1 = np.zeros((16))        # Hidden Layer 1
n2 = np.zeros((16))        # Hidden Layer 2
n3 = np.zeros((31))        # Output

c  = np.zeros((31))        # Cost / Error tiap node output

# Diff(Gradien) dari Weight dan Bias
w0G = np.zeros((31, 16))  # Input - Hidden layer 1
w1G = np.zeros((16, 16))   # Hidden Layer 1 - Hidden Layer 2
w2G = np.zeros((16, 31))   # Hidden Layer 2 - Output

b0G = np.zeros((16))       # Input - Hidden layer 1
b1G = np.zeros((16))       # Hidden Layer 1 - Hidden Layer 2
b2G = np.zeros((31))       # Hidden Layer 2 - Output

def insertRandomNum(array, namefile='data.dat'):
    if array.ndim == 2:
        dimX = array.shape[0]
        dimY = array.shape[1]
        for i in range(0, dimX):
            for j in range(0, dimY):
                array[i, j] = round(random.randrange(-1000, 1000, 1)/100, 2)
    else:
        dimX = array.shape[0]
        for i in range(0, dimX):
            array[i] = round(random.randrange(-1000, 1000, 1)/100, 2)
    np.savetxt(namefile, array)            

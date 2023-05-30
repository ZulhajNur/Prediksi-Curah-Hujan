import time
import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from threading import Thread
from multiprocessing import Process

layerInput  = np.zeros((1024))
layer1      = np.zeros((16))
layer2      = np.zeros((16))
layerOutput = np.zeros((10))

# Weight
w0 = np.zeros((1024, 16))  # Input - Hidden layer 1
w1 = np.zeros((16, 16))    # Hidden Layer 1 - Hidden Layer 2
w2 = np.zeros((16, 10))    # Hidden Layer 2 - Output

# Bias
b0 = np.zeros((16))        # Input - Hidden layer 1
b1 = np.zeros((16))        # Hidden Layer 1 - Hidden Layer 2
b2 = np.zeros((10))        # Hidden Layer 2 - Output

# Node dan Keluaran Activation Function
n0 = np.zeros((1024))      # Input
nx = np.zeros((1024))      # Input
n1 = np.zeros((16))        # Hidden Layer 1
n2 = np.zeros((16))        # Hidden Layer 2
n3 = np.zeros((10))        # Output

c  = np.zeros((10))        # Cost / Error tiap node output

# Diff(Gradien) dari Weight dan Bias
w0G = np.zeros((1024, 16)) # Input - Hidden layer 1
w1G = np.zeros((16, 16))   # Hidden Layer 1 - Hidden Layer 2
w2G = np.zeros((16, 10))   # Hidden Layer 2 - Output

b0G = np.zeros((16))       # Input - Hidden layer 1
b1G = np.zeros((16))       # Hidden Layer 1 - Hidden Layer 2
b2G = np.zeros((10))       # Hidden Layer 2 - Output

w0 = np.loadtxt('w0.dat')
w1 = np.loadtxt('w1.dat')
w2 = np.loadtxt('w2.dat')
b0 = np.loadtxt('b0.dat')
b1 = np.loadtxt('b1.dat')
b2 = np.loadtxt('b2.dat')

num = 0
cor = 0

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

class gradient():
    def gradWeight(func, learnRate, mode, aL_1, aL, wL, bL, y):
        if y == numReal:    # Jika angka tebakan benar
            y = 1
        else:               # Jika angka tebakan salah
            y = 0
        diff = -learnRate * (aL_1 * func(wL*aL_1+bL, mode)*2*(aL - y))
        return diff
        
    def gradBias  (func, learnRate, mode, aL_1, aL, wL, bL, y):
        if y == numReal:    # Jika angka tebakan benar
            y = 1
        else:               # Jika angka tebakan salah
            y = 0
        diff = -learnRate * (func(wL*aL_1+bL)*2*(aL - y))
        return diff

##count = 0
def backprop(mode, wArr, bArr, n1Arr, n0Arr, wGrad, wRowStart=None, wRowFinish=None, wColStart=None, wColFinish=None, learnRate=0.1):

    if wRowStart == None:
        wRowStart=0
    if wRowFinish == None:
        wRowFinish=wArr.shape[0]
    if wColStart == None:
        wColStart=0
    if wColFinish ==None:
        wColFinish=wArr.shape[1]

##    global count
    summ = wArr.shape[0]*wArr.shape[1]
    
    for wC in range(wColStart, wColFinish):
        for wR in range(wRowStart, wRowFinish):
            for b in range(0, bArr.shape[0]):
                for n1 in range(0, n1Arr.shape[0]):
                    for n0 in range(0, n0Arr.shape[0]):

                        if wC == b:
                            if wC == n1:
                                if wR == n0:
##                                    count += 1
                                    wGrad[wR, wC] = round(gradient.gradWeight(f1, learnRate, mode, n0Arr[n0], n1Arr[n1], wArr[wR, wC], bArr[b], numPred), 2)
##                                    print('{}/{}---{}.{}.{}.{}---{}|{}|{}|{}---{}'.format(count, summ,  n0, n1, (wR, wC), b,  n0Arr[n0], n1Arr[n1], wArr[wR, wC], bArr[b], w0G[wR, wC]))
##                                    print('{}/{}'.format(count, summ))
                                else:
                                    pass

def f(x, mode = 'linear'):
    if   mode == 'linear':
        return x
    elif mode == 'tanh':
        y = (np.e**x-np.e**-x)/(np.e**x+np.e**-x)
        return y
    elif mode == 'sigmoid': 
        y = 1/(1+np.e**-x)
        return y

def f1(x, mode = 'linear'):
    if   mode == 'linear':
        return 1
    elif mode == 'tanh':
        y = 4/(np.e**x+np.e**-x)**2
        return y
    elif mode == 'sigmoid':
        y = np.e**-x/((1+np.e**-x)**2)
        return y

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def actFunc(node, weight, bias):
    x = np.dot(weight, node) + bias
    return x

def normalize(x, maxx):
    y = (x)/(maxx)
    return y

def costFunc(x, t):
    y = (x - t)**2
    return y


### Command di GUI
def randomNum():
    # Mengambil tulisan angka acak
    global numReal, numVar, n0, n1, n2, n3, num, cor, cost, numPred
    global w0, w1, w2, b0, b1, b2
    
    num = num + 1
    label4_4.config(text='Simulation-{}'.format(num))
    
    numReal = random.randrange(0,10)
    numVar  = random.randrange(0,50)
    img     = Image.open('Number\{}\{}_{}.png'.format(numReal, numReal, numVar))
    imgShow = ImageTk.PhotoImage(img.resize((256, 256), resample=0))
    labelImg.config(image = imgShow)
    labelImg.image = imgShow
    labelRes.config(text='This is number {}'.format(numReal))
    
    imgArr = np.array(img)
    imgArr = rgb2gray(imgArr)
    imgArr = imgArr/255 # Normalisasi
    n0 = imgArr.ravel() # Memasukkan semua nilai pixel ke dalam Input(n0)

    # Perhitungan Activation Function untuk tiap Hidden Layer dan Output
    maxx = np.max(n0)
    
    global act_n1, act_n2, act_n3
    act_n1 = 'sigmoid'
    act_n2 = 'sigmoid'
    act_n3 = 'linear'
    
    for i in range(0, len(n0)):
        n0[i] = normalize(n0[i], maxx)
    for i in range(0, len(n0)):
        nx[i] = n0[i]

    for i in range(0, n1.shape[0]):
        n1[i] = f(actFunc(w0[:, i], n0, b0[i]), mode = act_n1)
    for i in range(0, n2.shape[0]):
        n2[i] = f(actFunc(w1[:, i], n1, b1[i]), mode = act_n2)
    for i in range(0, n3.shape[0]):
        n3[i] = f(actFunc(w2[:, i], n2, b2[i]), mode = act_n3)

    numPred = np.unravel_index(np.argmax(n3), n3.shape) # Memilih output dengan nilai tertinggi
    numPred = numPred[0]
    label4_1.config(text='This is number {}'.format(numPred))
    label4_2.config(text='The correct number is {}'.format(numReal))
    if numPred == numReal:
        label4_3.config(text='Recognition correct')
        cor = cor + 1
    else:
        label4_3.config(text='Recognition incorrect')
    label4_5.config(text='Recognition correct = {}/{}'.format(cor, num))
    
    n3Img = (n3/max(n3))*255
    
    img = ImageTk.PhotoImage(Image.fromarray(n3Img).resize((25, 256), resample=0))
    labelImgOut2.config(image = img)
    labelImgOut2.image = img
    
    # Menghitung Cost / Error
    for i in range(0, 10):
        if numPred == i:
            c[i] = costFunc(n3[i]/max(n3), 1)
##            print(i)
        else:
            c[i] = costFunc(n3[i]/max(n3), 0)
    
    cost = sum(c)/10
    label4_6.configure(text='Cost = {}'.format(cost))

##    backprop(act_n1, w0, b0, n1, n0, w0G)
##    backprop(act_n2, w1, b1, n2, n1, w1G)
##    backprop(act_n3, w2, b2, n3, n2, w2G)

    w0 = w0 + w0G
    w1 = w1 + w1G
    w2 = w2 + w2G

    # Saving New Weight Value
    np.savetxt('w0.dat', w0)
    np.savetxt('w1.dat', w1)
    np.savetxt('w2.dat', w2) 

##    print(w0G)
##    b0 += b0G
##    b1 += b1G
##    b2 += b2G

def simulation():
    global num, cor, x1, y1, x1a, x1b, y1a, y1b
    
##    while num < 1:
    randomNum()
    chart(num, cost)
    reloadChart()
    root.update()

##    num, cor = 0, 0

x1a = []
x1b = []
y1a = []
y1b = []

def chart(x2, y2):
    if y2 < 10:
        if numPred == numReal:
            y1b.append(y2)
            x1b.append(x2)
        else:
            y1a.append(y2)
            x1a.append(x2)
    else:
        if numPred == numReal:
            y1b.append(10)
            x1b.append(x2)
        else:
            y1a.append(10)
            x1a.append(x2)

def reloadChart():
    global cnvFrame, canvas, toolbar
    
    if canvas == None:
        pass
    else:
        canvas.get_tk_widget().destroy()
        plt.close('all')

    frameFig, ax = plt.subplots(figsize=(2.5, 1))
    
    ax.scatter(x1b, y1b, s=8, color='red')
    ax.scatter(x1a, y1a, s=2, color='gray')
            
    canvas = FigureCanvasTkAgg(frameFig, cnvFrame)
    canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
    
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
buttonIm  = tk.Button(frame1, text='Randomize Number', command=simulation)
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
label4_4  = tk.Label(frame4, text='Simulation-{}'.format(num))
label4_4.grid(row=3, sticky='W')
label4_5  = tk.Label(frame4, text='Recognition correct = {}/{}'.format(cor, num))
label4_5.grid(row=4, sticky='W')
label4_6  = tk.Label(frame4, text='Cost = {}'.format('0'))
label4_6.grid(row=5, sticky='W')
cnvFrame = tk.Frame(frame4)
cnvFrame.grid(row=6)

##root.mainloop()

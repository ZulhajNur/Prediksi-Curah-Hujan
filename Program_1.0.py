import time
import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import openpyxl as op

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


w0 = np.loadtxt('w0.dat')
w1 = np.loadtxt('w1.dat')
w2 = np.loadtxt('w2.dat')
b0 = np.loadtxt('b0.dat')
b1 = np.loadtxt('b1.dat')
b2 = np.loadtxt('b2.dat')

num = 0
cor = 0

class gradient():
    def gradWeight(func, learnRate, mode, aL_1, aL, wL, bL, y):
        y = 1
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

    summ = wArr.shape[0]*wArr.shape[1]
    
    for wC in range(wColStart, wColFinish):
        for wR in range(wRowStart, wRowFinish):
            for b in range(0, bArr.shape[0]):
                for n1 in range(0, n1Arr.shape[0]):
                    for n0 in range(0, n0Arr.shape[0]):

                        if wC == b:
                            if wC == n1:
                                if wR == n0:
                                    wGrad[wR, wC] = round(gradient.gradWeight(f1, learnRate, mode, n0Arr[n0], n1Arr[n1], wArr[wR, wC], bArr[b], numPred), 2)
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

def actFunc(node, weight, bias):
    x = np.dot(weight, node) + bias
    return x

def normalize(x, maxx=None):
    if max != None:
        y = (x)/(maxx)
    else:
        y = x
    return y

def normalize2(x, xmin, xmax, maxVal):
    y = (x-xmin)/(xmax-xmin) * maxVal
    return y

def costFunc(x, t):
    y = (x - t)**2
    return y

def klasifikasi(x, i):
    if x[i] >= 30:
        x[i] = 'Hujan Lebat'
    elif x[i] <= 20:
        x[i] = 'Tidak Hujan'
    else:
        x[i] = 'Hujan Gerimis'

### Command di GUI

ch1 = []
ch2 = []

workbook = op.load_workbook('CHRR.xlsx')
worksheet = workbook['2021']

def randomNum():
    # Mengambil tulisan angka acak
    global numReal, numVar, n0, n1, n2, n3, num, cor, cost, ch1, ch2
    global w0, w1, w2, b0, b1, b2

    numReal = random.randrange(0,10)
    numVar  = random.randrange(0,31)

    for i in worksheet['F2':'F32']:
        for cell in i:
            ch1.append(cell.value)
    ch1 = np.array(ch1)
    n0 = np.array(ch1)

    for i in worksheet['G2':'G32']:
        for cell in i:
            ch2.append(cell.value)
    ch2 = np.array(ch2)

    for i in range(0,31):
        if ch1[i] == None:
            ch1[i] = 0
        if ch2[i] == None:
            ch2[i] = 0

        if ch1[i] >= 8888:
            ch1[i] = 0
        if ch2[i] >= 8888:
            ch2[i] = 0
    
    # Perhitungan Activation Function untuk tiap Hidden Layer dan Output
    maxx = np.max(n0)
##    maxx = 
    
    global act_n1, act_n2, act_n3
    act_n1 = 'linear'
    act_n2 = 'sigmoid'
    act_n3 = 'linear'
    
    for i in range(0, len(n0)):
        n0[i] = normalize(n0[i], maxx = maxx)
    for i in range(0, len(n0)):
        nx[i] = n0[i]

    for i in range(0, n1.shape[0]):
        n1[i] = f(actFunc(w0[:, i], n0, b0[i]), mode = act_n1)
    for i in range(0, n2.shape[0]):
        n2[i] = f(actFunc(w1[:, i], n1, b1[i]), mode = act_n2)
    for i in range(0, n3.shape[0]):
        n3[i] = f(actFunc(w2[:, i], n2, b2[i]), mode = act_n3)
    for i in range(0, n3.shape[0]):
        n3[i] = normalize2(n3[i], np.min(n3), np.max(n3), (np.max(n3)-np.min(n3)))
    
    
##    # Menghitung Cost / Error
##    for i in range(0, 10):
##        if numPred == i:
##            c[i] = costFunc(n3[i]/max(n3), 1)
##        else:
##            c[i] = costFunc(n3[i]/max(n3), 0)

##    cost = sum(c)/10
##    label4_6.configure(text='Cost = {}'.format(cost))

##    backprop(act_n1, w0, b0, n1, n0, w0G)
##    backprop(act_n2, w1, b1, n2, n1, w1G)
##    backprop(act_n3, w2, b2, n3, n2, w2G)
##    w0 = w0 + w0G
##    w1 = w1 + w1G
##    w2 = w2 + w2G
##
##    # Saving New Weight Value
##    np.savetxt('w0.dat', w0)
##    np.savetxt('w1.dat', w1)
##    np.savetxt('w2.dat', w2) 
##    print(w0G)
##    b0 += b0G
##    b1 += b1G
##    b2 += b2G

def simulation():
    global num, cor, x1, y1
    randomNum()

simulation()

##fig, ax = plt.subplots()


zzz = list(n3)
for i in range(0, 31):
    klasifikasi(zzz, i)

print(zzz)

plt.bar(range(1, 32), (abs(n3-ch2)))

##ax.bar(np.arange(1, 32)+0.00, n3, width = 0.25)       # Hasil Prediksi                    (Biru)
##ax.bar(np.arange(1, 32)+0.25, ch1, width = 0.25)      # Curah Hujan Data Training         (Oren)
##ax.bar(np.arange(1, 32)+0.50, ch2, width = 0.25)      # Curah Hujan Perbandingan Prediksi (Hijau)
plt.show()


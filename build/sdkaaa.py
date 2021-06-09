import numpy as np
from scipy.fftpack import fft, fftshift, ifft, ifft2
import math
from scipy.interpolate import interp1d
import soundfile
from matplotlib import pyplot as plt

def testfunc():
    print("python test!!!!!!!!!!!!!!!!!!!!!!!!")
    [Aux_data,FS] = soundfile.read('record.wav')
    trig = Aux_data[:,1] # %-1*T(:,1) #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    s = Aux_data[:,0] # %-1*S(:,1)
    nummax=min(s.shape[0],trig.shape[0])
    pi = 3.14159265
    c = 3E8 #(m/s) speed of light
    V = 2 #(m/s) speed of UAV
    # %radar parameters
    Tp = 10E-3 # (s) pulse time%%%%%%%%%%%%%%%%%%%%%实际三角波的周期是20ms!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Trp = 0.35 # %(s) min range profile time duration
    N = int(Tp*FS) # %# of samples per pulse
    Nrp = Trp*FS # %min # samples between range profiles 距离像    采集的点数
    fstart = 2420E6 # %(Hz) LFM start frequency 2260
    fstop = 2480E6 # %(Hz) LFM stop frequency   2590%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!
    #%BW = fstop-fstart # %(Hz) transmti bandwidth
    #%f = linspace(fstart, fstop, N/2) # %instantaneous transmit frequency  瞬时发射频率


    pzero = (trig>0.01) # %%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!根据实际电压改动
    aa = np.sum(pzero-1) 
    k=1
    first_period=1
    # jj=1
    SIF = np.zeros((N,1))
    #ineed=np.zeros((1,nummax))
    # ineed = np.zeros(nummax)
    sif = None
    while(k<100):
        print("loop\n")
        k=k+1
    print("python test!!!!!!!!!!!!!!!!!!!!!!!!eeeeeeeeeeeeeeeeeeeeeeeeeeeeend")



import numpy as np
from scipy.fftpack import fft, fftshift, ifft, ifft2
import math
from scipy.interpolate import interp1d
import soundfile
from matplotlib import pyplot as plt
from PIL import Image


#  %wk 连续飞行 实测代码
# %-------------------------------------------%
# %Process raw data here
# clear all;
# close all;
# read the raw data .m4a file here
# [T1,FS] = soundfile.read('trigger_move.wav')
# [S1,FS] = soundfile.read('signal_move.wav')
# T1 = np.ones((10000,2))
# S1 = np.ones((10000,2))
def testfunc():
    [Aux_data,FS] = soundfile.read('./record_0.wav')

    # simulated signal
    trig = Aux_data[:,1] # %-1*T(:,1) #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    s = Aux_data[:,0] # %-1*S(:,1) 

    # s = s[425100:len(s)]
    # trig = trig[690800:len(trig)]  #%265700+425100
    nummax=min(s.shape[0],trig.shape[0])
    print(nummax)
    # num1=1  #%300000 
    # num2=int(1e6)  #%nummax%%%%%%%%%%%%%%%%%%%%%%%%%%%%%！！！！！！！！！！！！！！！！
    # t=np.linspace(num1,num2+1,num2)  #%size(Y,1)
    # trig1=trig[num1:num2]
    # s1=s[num1:num2]
    # % 
    # figure(1);
    # plot(t,trig1,'r');%%%%%%%方波  红
    # hold on
    # plot(t,s1,'b');%%%%%%%%%%信号  蓝


    # %constants
    pi = 3.14159265
    c = 3E8 #(m/s) speed of light
    V = 5 #(m/s) speed of UAV
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
    # k=1
    # first_period=1
    # # jj=1
    # SIF = np.zeros((N,1))
    # #ineed=np.zeros((1,nummax))
    # # ineed = np.zeros(nummax)
    # sif_counter = 0
    # while (k<nummax-N+1):
    #     if (pzero[int(k)]==1) and (pzero[int(k-1)]==0):
    #         if(first_period):
    #             k = k+1
    #             first_period = 0
    #         else:
    #             k = k+N/4
    #             sif_counter += 1

    sif = None
    k=1
    first_period = 1
    cc = 0
    while (k< nummax-N+1):
        cc += 1
        if (pzero[int(k)]==1) and (pzero[int(k-1)]==0):
            if(first_period): #%一般第一个方波周期不完整，故舍弃
                k=k+1
                first_period=0
            else:
                # ineed[int(k):int(k+N-1)]=s[int(k):int(k+N-1)]
                SIF = s[int(k):int(k+N-1)]                              
                k=k+N/4      
                # %%%%%%%%%% hilbert transform 希尔伯特变换
                q = ifft(SIF)
                tmp_arr = fft(q[int((q.shape[0])/2+1):q.shape[0]])
                for i_idx in range(len(tmp_arr)):
                        if np.isnan(tmp_arr[i_idx])==1:
                            tmp_arr[i_idx] = 1E-30
                if(sif is None):
                    sif = tmp_arr
                else:
                    sif = np.row_stack((sif,tmp_arr))  
                # sif[jj,:] = fft(q[(q.shape[0])/2+1:q.shape[0]])         
                # %%%%%%%%%%

                # jj = jj+1
            # % SIF(:,1)=0
        else:
            k=k+1
    print(cc)


    # %load IQ converted data here
    sif_mean = np.mean(sif,0)
    for ii in range(sif.shape[0]):
        sif[ii,:] = sif[ii,:] - sif_mean

    # sif = s 
    # %image without background subtraction
    # clear s
    # clear sif_sub

    # %***********************************************************************
    # %radar parameters

    fc = (fstop-fstart)/2 + fstart # %(Hz) center radar frequency(2590E6 - 2260E6)/2 + 2260E6
    B = fstop-fstart # %(hz) bandwidth(2590E6 - 2260E6)
    Rs = 40                          #height!!!!!!!!!!!!!!!!
    Xa = 0 # %(m) beginning of new aperture length
    delta_x =(Tp*2)*V #%2*(1/12)*0.3048; %(m) 2 inch antenna spacing  5cm=0.969inch
    L = delta_x*(sif.shape[0]) # %(m) aperture length

    Xa = np.linspace(-L/2, L/2, int(L/delta_x)) # %(m) cross range position of radar on aperture L
    Za = 0
    Ya = Rs # %THIS IS VERY IMPORTANT, SEE GEOMETRY FIGURE 10.6
    t = np.linspace(0, Tp, int(sif.shape[1])) # %(s) fast time, CHECK SAMPLE RATE
    Kr = np.linspace(((4*pi/c)*(fc - B/2)), ((4*pi/c)*(fc + B/2)), (sif.shape[1])) #%%Kr=4pi/lamda

    S_image = 20*np.log10(abs(fftshift(fft(sif, None, 1), 1)))
    # imagesc(np.linspace(-FS/2,FS/2,S_image.shape[1]), Xa, S_image)
    X,Y = np.meshgrid(np.linspace(-FS/2,FS/2,S_image.shape[1]), Xa)
    plt.pcolormesh(np.linspace(-FS/2,FS/2,S_image.shape[1]), Xa, S_image, vmin=S_image.max()-50,vmax=S_image.max())
    # colormap(gray)
    plt.title('Magnitude of 1-D FFT of Downrange Data')
    plt.xlabel('f(Hz)')
    plt.ylabel('K_x (rad/m)')
    # cbar = colorbar
    # set(get(cbar, 'Title'), 'String', 'dB','fontsize',13)
    
    # plt.savefig("1dfft2.jpg")
    plt.savefig("../../../../../api_sample/camera_media_emu/media_file/1dfft.jpg")
    plt.figure()
    plt.plot(range(6000), Aux_data[:6000,0])
    # plt.savefig("waveform.jpg")
    plt.savefig("../../../../../api_sample/camera_media_emu/media_file/waveform.jpg")
    
    plt.figure()
    plt.plot(range(6000), Aux_data[:6000,1])
    # plt.savefig("square.jpg")
    plt.savefig("../../../../../api_sample/camera_media_emu/media_file/square.jpg")

    print("Finished")
# testfunc()


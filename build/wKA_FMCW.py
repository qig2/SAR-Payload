import numpy as np
from scipy.fftpack import fft, fftshift, ifft, ifft2
import math
from scipy.interpolate import interp1d
import soundfile
from matplotlib import pyplot as plt


def testfunc():
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
    [Aux_data,FS] = soundfile.read('record.wav')

    # simulated signal

    trig = Aux_data[:,1] # %-1*T(:,1) #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    s = Aux_data[:,0] # %-1*S(:,1) 
    # s = s[425100:len(s)]
    # trig = trig[690800:len(trig)]  #%265700+425100
    nummax=min(s.shape[0],trig.shape[0])
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
    while (k< nummax-N+1):
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


    # ineed1 = ineed[num1:num2]

    #hold on
    # plot(t,ineed1,'g');%%%%%%%%%%信号  绿

    # for i in sif:
    #     for k in i:
    #         if np.isnan(k)==1:
    #             k = 1E-30
    #%set all Nan values to 0

    # %SAR data should be ready here
    # clear s

    # s = sif

    # %-------------------------------------------%
    # %load additional varaibles and setup constants for radar here
    # %clear all;

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
    Rs = 0
    Xa = 0 # %(m) beginning of new aperture length
    delta_x =(Tp*2)*V #%2*(1/12)*0.3048; %(m) 2 inch antenna spacing  5cm=0.969inch
    L = delta_x*(sif.shape[0]) # %(m) aperture length
    print(L)
    Xa = np.linspace(-L/2, L/2, int(L/delta_x)) # %(m) cross range position of radar on aperture L
    Za = 0
    Ya = Rs # %THIS IS VERY IMPORTANT, SEE GEOMETRY FIGURE 10.6
    t = np.linspace(0, Tp, int(sif.shape[1])) # %(s) fast time, CHECK SAMPLE RATE
    Kr = np.linspace(((4*pi/c)*(fc - B/2)), ((4*pi/c)*(fc + B/2)), (sif.shape[1])) #%%Kr=4pi/lamda

    # %%
    # %load sif_fence_bicycle;
    # %apply hanning window to data first  应用汉宁窗
    # %加窗为了更好地符合FFT处理的周期性要求，防止频谱泄漏

    N = sif.shape[1]
    i_tmp = np.linspace(1,N+1,N)
    H = 0.5+0.5*np.cos(2*pi*(i_tmp-N/2)/N)


    for ii in range(sif.shape[0]):
        # if (sif_h is None):
        #     sif_h = sif[ii,:]*H
        # else:
        #     sif_h = np.row_stack((sif_h,sif[ii,:]*H))
        # sif_h[ii,:] = sif[ii,:]*H   #////dot product????????
        sif[ii,:] = sif[ii,:]*H
    # sif = sif_h

    zpad = sif.shape[0]
    # zpad = 1024 # %cross range symetrical zero pad%%%%%%%%%%%%%%%%可以调整成像宽度
    # while (sif.shape[0]>zpad):
    #     zpad += 1024
    # szeros = np.zeros((zpad, sif.shape[1]), dtype=np.complex)

    # for ii in range(sif.shape[1]): 
    #     index = round((zpad - sif.shape[0])/2)
    #     szeros[index:(index + sif.shape[0]),ii] = sif[:,ii] # %symetrical zero pad

    # sif = szeros
    # clear ii index szeros

    S = fftshift(fft(sif, None, 0), 0)
    # clear sif
    Kx = np.linspace((-pi/delta_x), (pi/delta_x), (S.shape[0]))  #  %(2pi/delta_x)分成2048份,相当于2pi/lmd,delta_x为方位向采样频率对应的步长

    # %matched filter
    aa_max=10
    aa_min=1e3
    # %create the matched filter eq 10.8
    phi_mf = np.zeros((S.shape[0],S.shape[1]))
    # Krr = np.zeros((S.shape[0],S.shape[1]))
    # Kxx = np.zeros((S.shape[0],S.shape[1]))
    for ii in range(S.shape[1]):  # 1:size(S,2):  # %step thru each time step row to find phi_if   441
        for jj in range(S.shape[0]):  #= 1:size(S,1)      # %step through each cross range in the current time step row  2048
            #%phi_mf(jj,ii) = -Rs*Kr(ii) + Rs*sqrt((Kr(ii))^2 - (Kx(jj))^2);
            aa=np.sqrt((Kr[ii])**2 - (Kx[jj])**2)  #%aa即Ky
            if aa>aa_max:
                aa_max=aa
            
            if aa<aa_min:
                aa_min=aa
            
            phi_mf[jj,ii] = Rs*np.sqrt((Kr[ii])**2 - (Kx[jj])**2)
            # Krr[jj,ii] = Kr[ii] # %generate 2d Kr for plotting purposes
            # Kxx[jj,ii] = Kx[jj]  # %generate 2d Kx for plotting purposes


    smf = np.exp(1j*phi_mf) # %%%%%%%%%%%%

    # %note, we are in the Kx and Kr domain, thus our convention is S_mf(Kx,Kr)

    # %appsly matched filter to S
    S_mf = S*smf

    # clear smf phi_mf;
    # %**********************************************************************
    # %perform the Stolt interpolation

    # %FOR DATA ANALYSIS
    # %kstart、kstop分别为sqrt(Kr^2-Kx^2)的最小值和最大值
    kstart =71.7 # %73
    kstop = 108.5
    # % kstart =aa_min;
    # % kstop = aa_max;

    Ky_even = np.linspace(kstart, kstop, 1024) # %create evenly spaced Ky for interp for real data

    # clear Ky S_St


    # %for ii = 1:size(Kx,2)
    S_st = np.zeros((zpad,len(Ky_even)),dtype=complex)
    Ky = np.zeros((zpad,len(Kr)))
    # count = 0
    for ii in range(zpad): 
    # %for ii = round(.2*zpad):round((1-.2)*zpad)
        Ky[ii,:] = np.sqrt(Kr**2-Kx[ii]**2)
        tmp_func = interp1d(Ky[ii,:], S_mf[ii,:], bounds_error=False, fill_value=1E-30)
        S_st[ii,:] = tmp_func(Ky_even)
        #Ky[count,:] = np.sqrt(Kr**2 - Kx[ii]**2)
        # if(count==0):
        #     Ky = np.sqrt(Kr**2-Kx[ii]**2)
        #     tmp_func = interp1d(Ky, S_mf[ii,:], bounds_error=False, fill_value=1E-30)
        #     S_st[ii,:] = tmp_func(Ky_even)
        # else:
        #     Ky = np.row_stack((Ky,np.sqrt(Kr**2-Kx[ii]**2)))
        # # %S_st(ii,:) = (interp1(Ky(ii,:), S_mf(ii,:), Ky_even)).*H;
        #     tmp_func = interp1d(Ky[count,:], S_mf[ii,:],bounds_error=False, fill_value=1E-30)
        #     S_st = np.row_stack((S_st,tmp_func(Ky_even)))
        # # S_st[count,:] = (interp1d(Ky[count,:], S_mf[ii,:], Ky_even))  #%yy=interp1(x,y,xx)
        # count = count + 1

    # print(S_st.shape)
    # S_st(find(isnan(S_st))) = 1E-30 # %set all Nan values to 0
    # for i in range(S_st.shape[0]):
    #     for j in range(S_st.shape[1]):
    #         if(math.isnan(S_st[i][j])):
    #             S_st[i][j] = 1E-30

    # clear S_mf ii Ky

    # N = len(Ky_even)
    # for ii = 1:N
    #     H(ii) = 0.5 + 0.5*cos(2*pi*(ii-N/2)/N);
    # end
    # i_tmp = np.linspace(1,N+1,N)
    # H = 0.5+0.5*np.cos(2*pi*(i_tmp-N/2)/N)

    # S_sth = None
    # for ii in range(S_st.shape[0]):
    #     if S_sth is None:
    #         S_sth = S_st[ii,:]*H
    #     else:
    #         S_sth = np.row_stack((S_sth,S_st[ii,:]*H))
    #     # S_sth[ii,:] = S_st[ii,:]*H   #////What's the reason for calculating S_sth???????




    # %*********************************************************************
    # %perform the inverse FFT's
    # %new notation:  v(x,y), where x is crossrange
    # %first in the range dimmension

    # clear v Kr Krr Kxx Ky_even

    v = ifft2(S_st,(S_st.shape[0]*4,S_st.shape[1]*4))   # %2维ifft,X=ifft2(Y,m,n)将Y维度填充到m*n,X也为m*n

    # %bw = (3E8/(4*pi))*(max(xx)-min(xx));

    bw = 3E8*(kstop-kstart)/(4*pi)   # %c/lamda=f
    max_range = (3E8*S_st.shape[1]/(2*bw))*1/.3048    # %  573  !!!!!!!!!!!!!!!!!!!!!!
    # figure(3);
    S_image = v    # %edited to scale range to d^3/2   4096*8192
    S_image = np.fliplr(np.rot90(S_image))    #  %先将S_image逆时针旋转90度，再沿水平方向翻转，整体效果相当于对矩阵进行中心对称变换
    cr1 = -L/2/0.3048    #%-80; %(ft) cross range %%%%%%%选取感兴趣的成像宽度
    cr2 = L/2/0.3048    #%80; %(ft)
    dr1 = 1      #%1 + Rs/.3048; %(ft) down range
    dr2 = 500 + Rs/.3048     # %(ft)   %%%%%%%%%%%%%选取感兴趣的成像长度
    # %data truncation 数据截断
    dr_index1 = round((dr1/max_range)*(S_image.shape[0]))    # % size(S_image,1),取矩阵行数是因为上面进行了中心对称，原来矩阵横向对应距离向，现在纵向对应距离向
    dr_index2 = round((dr2/max_range)*(S_image.shape[0]))    # % roung(350/573*4096)=2499

    cr_index1 = round(((cr1+zpad*delta_x/(2*.3048))/(zpad*delta_x/.3048))*(S_image.shape[1]))     #%!!!!!!!!!!!!!!!!!!!!!!
    cr_index2 = round(((cr2+zpad*delta_x/(2*.3048))/(zpad*delta_x/.3048))*(S_image.shape[1]))
    # % cr_index1:cr_index2的范围 是在cr1:cr2的范围上左右各加1/2的zpad视野
    # %round( ( (80+2048*0.05/2*0.3048) / (2048*0.05/0.3048) ) *8192   )=6016

    trunc_image = S_image[dr_index1:dr_index2,cr_index1:cr_index2]   # % 2493*3841
    downrange = np.linspace(-1*dr1,-1*dr2, trunc_image.shape[0]) + Rs   # %/.3048 
    crossrange = np.linspace(cr1, cr2, trunc_image.shape[1]) 
    # %scale down range columns by range^(3/2), delete to make like
    # %dissertation again

    # clear ii 

    for ii in range(trunc_image.shape[1]):  #= 1:size(trunc_image,2): 
        trunc_image[:,ii] = (trunc_image[:,ii].T)*(abs(downrange*.3048))**(3/2)  # %把矩阵数值扩大到原来的1.5次方

    trunc_image = 20*np.log10(abs(trunc_image))  # %added to scale to d^3/2
    # print(trunc_image.max())
    # max(max(trunc_image))
    #imagesc(crossrange, downrange, trunc_image, [max(max(trunc_image))-50, max(max(trunc_image))-0]); 
    X,Y = np.meshgrid(crossrange,downrange) 
    plt.pcolormesh(crossrange,downrange, trunc_image, vmin=trunc_image.max()-50,vmax=trunc_image.max())
    plt.gca().invert_yaxis()

    plt.ylabel('Downrange (ft)')
    plt.xlabel('Crossrange (ft)')
    plt.savefig("../../../../../api_sample/camera_media_emu/media_file/test_pic.png")
    # colormap('default');
    # title('fence(bicycle)')
    # ylabel('Downrange (ft)');
    # xlabel('Crossrange (ft)');
    # axis equal;
    # cbar = colorbar;
    # set(get(cbar, 'Title'), 'String', 'dB','fontsize',13); 
    # print(gcf, '-djpeg100', 'fence_bicycle.jpg');

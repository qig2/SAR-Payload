import numpy as np
import scipy
from scipy.fftpack import fft, fftshift, ifft
import math


def SARimage_python():
    pi = 3.1415926
    c = 3e+8                                    
    R0 = 150                                    
    V = 10                                      
    PRT = 10e-3                                 
    dert_f = 60e+6                              
    Fc = 2.45e+9                                
    Fr = 48e+3                                  
    theta_B=50*pi/180                           
    theta_sqc = 0*pi/180                        


    lmd=c/Fc                                    
    Fa = 1/PRT                                  #
    Lsar=R0*theta_B                             #
    Tsar=Lsar/V                                 #
    Kr=dert_f/PRT                               #


    Nslow=math.ceil(Fa*Tsar*1.2)                        #
    M=Nslow 
    u=np.linspace(-Tsar/2*1.2,Tsar/2*1.2,Nslow).reshape(1,-1)      #

    Nfast=math.ceil(PRT*Fr)                             #
    N=Nfast 
    t=np.linspace(0,PRT,Nfast).reshape(1,-1)                        #

    Dslow = np.dot(np.ones([N,1]),u)                              #
    Dfast = np.dot(t.T,np.ones([1,M]))                           #
    F_fr = np.linspace(0,Fr,Nfast).reshape(1,-1)                      #
    F_fa = np.linspace(-Fa/2,Fa/2,Nslow).reshape(1,-1)                #
    FA=np.dot(np.ones([N,1]),F_fa)                              #
    FR=np.dot(F_fr.T,np.ones([1,M]))                             #

    Rref=0                                         #
    x0=20                                          #
    D=(Dslow+Dfast)*V-x0 

    R=np.sqrt(D*D+R0*R0-2*R0*np.sin(theta_sqc)*D)        #

    Fd=2*V/lmd                                     #

    tao=2*R/c                                      #

    t_tempt=Dfast-2*R/c                            #
    t_tempt[(t_tempt<=0) | (t_tempt>=PRT)] = 0
    t_tempt[(0<t_tempt) & (t_tempt<PRT)] = 1

    # Sr=exp(j*2*pi*(Kr.*tao-Fd).*Dfast).*exp(j*2*pi*Fc.*tao).*exp(j*2*pi*Kr/2.*(tao).^2).*(0<t_tempt&t_tempt<PRT) 
    Sr=np.exp((-1j)*4*pi*R/lmd)*np.exp(1j*2*pi*Fd*Dfast)*np.exp(-1j*4*pi*Kr/c*(Dfast-2*Rref/c)*(R-Rref))*np.exp(1j*4*pi*Kr/c**2*(R-Rref)**2)*t_tempt
    # (0<t_tempt and t_tempt<PRT)
                                                   #
    # aa=4*pi*Kr/c*(Dfast-2*Rref/c)*(R-Rref) 
    # aaaa=max(aa) 



    H_FdC=np.exp(-1j*2*pi*Fd*Dfast)                   #
    Sdwf=Sr*H_FdC                                 #


    SdwfFa=fftshift(fft(Sdwf,None,axis=1),1) 


    H_DFS_RMC_SRC=np.exp(-1j*2*pi*(FA+Fd*np.sin(theta_sqc))*Dfast)*np.exp(1j*4*pi*Kr/c*R0*lmd**2*FA**2*(Dfast-2*Rref/c)/(8*V**2*np.cos(theta_sqc)**2))*np.exp(1j*2*pi*lmd*Kr**2*R0/c**2*(-lmd*lmd*FA*FA/(4*V*V*np.cos(theta_sqc)**2))/(1-lmd**2*FA**2/(4*V**2*np.cos(theta_sqc)**2))**(3/2)*(Dfast-2*Rref/c)**2) 
    SFa=SdwfFa*H_DFS_RMC_SRC 


    #H_AZI_REF=exp(j*2*pi*V*V*(Fr-4*Kr*Rref/c)/(c*R0).*(Dslow).^2).*exp(-j*pi*((Fa).^2)*R0*c/(2*V*V*(Fc-4*Kr*Rref/c))) 

    H_AZI_REF=np.exp(1j*4*pi/lmd*R0*np.sqrt(1-lmd**2*FA**2/(4*V**2*np.cos(theta_sqc)**2)))*np.exp(-1j*2*pi*np.sin(theta_sqc)*R0*FA**3/(lmd*np.cos(theta_sqc)*((2*V/lmd*(1))**2-(FA)**2)**(3/2))) 
    H_PMF=np.exp(1j*4*pi*Rref/c*FR)                    #
    SFaFr=fft(SFa,None,axis=0)                             #
    SFF=SFaFr*H_AZI_REF*H_PMF 
    SFr=ifft(fftshift(SFF,1),None,axis=1)                 #
    #F=linspace(0,Fr,Nfast)'; Gr=abs(SFr); figure; colormap(gray); 
    #imagesc(u*V,F,255-Gr);                          
    print("py file end\n")


SARimage_python()
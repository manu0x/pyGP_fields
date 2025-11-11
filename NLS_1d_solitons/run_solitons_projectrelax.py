import os
import sys
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve   
from scipy.optimize import elementwise

cwd = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)
print("CWD:",cwd)



from GPE import ImEx

from soliton_functions_projectrelax import *



def laplacian(u,xi2):
     ft_u = -np.fft.fft(u,u.shape)*xi2
     #ft_u[0,0] = 0.0
     return np.fft.ifft(ft_u,u.shape)



if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

   

    
    imex_sch = str(sys.argv[1])
    relax = int(sys.argv[2])  ## 0- No relax, 1- mass only relax, 2- full relax
    n_soliton =int(sys.argv[3])
    kin_type = str(sys.argv[4])  ## "B" or "Ek2"
    print("Command line args:",imex_sch,relax,n_soliton,kin_type)
    A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages = choose_imex(imex_sch)  
 

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex,emb_B=b_hat,im_C=C,ex_C=C)


    # List of tau values for which u wanna run simulation
    
    
    frame_dict_list = []
  

    t_list = []
    if n_soliton==2:
        kppa = 8.0
    elif n_soliton==3:  
        kppa = 18.0

    lap_fac = 1.0  ## Laplacian factor in front of kinetic energy term

    dt= 0.01     ## Choose dt
    t_ini = 0.0
    T = 50.0
    

    m = 1023  ## Number of grid points
    xL = -35.0; xR = 35.0; L = xR-xL
   
    x = np.arange(-m/2,m/2)*(L/m)


    xi = np.fft.fftfreq(m)*m*2*np.pi/L





    
    case="imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    if kin_type=="Ek1":
        save_dir = "./_data_projRel_Ek1/"
    elif kin_type=="Ek2":
        save_dir = "./_data_projRel_Ek2/"
    else:
        print("Kinetic type not recognized")
        exit()   
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    save_dir = save_dir+"/"+case
   
    
    def soliton_sol(t,x,kappa,n_soliton=n_soliton):
     
        if n_soliton == 2:
            sechx = 1./np.cosh(x)
            ut = np.exp(1j*t)*sechx*( 1+(3/4)*sechx**2*(np.exp(8*1j*t)-1) )/( 1-(3/4)*sechx**4*np.sin(4*t)**2 )
        elif n_soliton == 3:
            ut = (2*(3*np.exp(t*25*1j)*np.exp(x) + 15*np.exp(t*9*1j)*np.exp(9*x) + 48*np.exp(t*25*1j)*np.exp(7*x) + 48*np.exp(t*25*1j)*np.exp(11*x) + 24*np.exp(t*33*1j)*np.exp(3*x) + 54*np.exp(t*33*1j)*np.exp(5*x) + 3*np.exp(t*25*1j)*np.exp(17*x) + 54*np.exp(t*33*1j)*np.exp(13*x) + 24*np.exp(t*33*1j)*np.exp(15*x) + 135*np.exp(t*41*1j)*np.exp(9*x) + 30*np.exp(t*49*1j)*np.exp(5*x) + 120*np.exp(t*49*1j)*np.exp(7*x) + 120*np.exp(t*49*1j)*np.exp(11*x) + 30*np.exp(t*49*1j)*np.exp(13*x) + 60*np.exp(t*57*1j)*np.exp(9*x)))/(3*(np.exp(t*24*1j) + 10*np.exp(6*x) + 10*np.exp(12*x) + 45*np.exp(t*8*1j)*np.exp(8*x) + 45*np.exp(t*8*1j)*np.exp(10*x) + 18*np.exp(t*16*1j)*np.exp(4*x) + 9*np.exp(t*24*1j)*np.exp(2*x) + 18*np.exp(t*16*1j)*np.exp(14*x) + 64*np.exp(t*24*1j)*np.exp(6*x) + 36*np.exp(t*24*1j)*np.exp(8*x) + 36*np.exp(t*24*1j)*np.exp(10*x) + 64*np.exp(t*24*1j)*np.exp(12*x) + 18*np.exp(t*32*1j)*np.exp(4*x) + 9*np.exp(t*24*1j)*np.exp(16*x) + np.exp(t*24*1j)*np.exp(18*x) + 18*np.exp(t*32*1j)*np.exp(14*x) + 45*np.exp(t*40*1j)*np.exp(8*x) + 45*np.exp(t*40*1j)*np.exp(10*x) + 10*np.exp(t*48*1j)*np.exp(6*x) + 10*np.exp(t*48*1j)*np.exp(12*x))) 
    
        return ut

    
    q0_ini  = soliton_sol(t_ini,x,kppa,n_soliton)
    
  
    q0_ini_mass = np.mean(np.square(np.abs(q0_ini)))
    print("Initial mass is",q0_ini_mass,L)
 
    print("q0_ini",q0_ini.shape,q0_ini.dtype)

    u_ini = q0_ini

  

    print("Running solitons with scheme ",imex_sch)
   
    if relax>0:
        if relax==1:
            save_dir_case = save_dir+"_relaxed_massOnly"
        else:
            save_dir_case = save_dir+"_relaxed"
    else:
        save_dir_case = save_dir
    if not(os.path.exists(save_dir_case)):
            os.makedirs(save_dir_case)
    frm,tt,inv_change_dict,mass_l,mass_err_l,energy_err_l,E_l,err_l = run_SP_2d_example_1(dt,[x],[xi],kppa,t_ini,T,L,imx,[],u_ini,kin_type=kin_type,exact_soln_np=soliton_sol,dx_soln_jx=None,\
                                                            relax=relax,log_errs=True,lap_fac=lap_fac,num_plots=100,p=None,data_dir=save_dir_case)
    case_dict={"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"x":[x],"xi":[xi],\
                   "kappa":kppa,"dt":dt,"m":m,"mass_l":mass_l,"mass_err_l":mass_err_l,"energy_err_l":energy_err_l,"energy_l":E_l,"err_l":err_l}
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
            pickle.dump(case_dict,f)
    


    
    
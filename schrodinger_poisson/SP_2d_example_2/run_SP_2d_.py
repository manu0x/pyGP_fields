import os
import sys
import pickle
import jax
import jax.numpy as jnp
import numpy as np

cwd = "/".join(os.getcwd().split("/")[:-2])
sys.path.append(cwd)
print("CWD:",cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from SP_2d_functions import *





if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

   

   
    imex_sch = str(sys.argv[1])
    A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages = choose_imex(imex_sch)  
    print("C",C)

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex,emb_B=b_hat,im_C=C,ex_C=C)


    # List of tau values for which u wanna run simulation
    
    
    frame_dict_list = []
  

    t_list = []

    alpha = 5.0
    beta = 5.0
    epsilon = 1.0
    
    kppa = beta/(epsilon*alpha)  ## Coefficient in front of non-linear term
    non_linear_p = 5.0
    lap_fac = epsilon/(2.0*alpha*alpha)  ## Coefficient in front of laplacian term
    
    dt=1e-3     ## Choose dt
    t_ini = 0.0
    T = 30.0
    

    m = 100  ## Number of grid points
    xL = -1.0; xR = 1.0; L = xR-xL
    x_1d = np.arange(-m/2,m/2)*(L/m)
    x,y = np.meshgrid(x_1d,x_1d)

    xi = np.fft.fftfreq(m)*m*2*np.pi/L
    xix,xiy = np.meshgrid(xi,xi)

    
    inv_list = [mass,energy]

    
    case="imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    save_dir = "./_data/"
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    save_dir = save_dir+"/"+case
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    
    
    
    
    

    def initial_conditions(x):
        #N = len(x)
        ut = (np.sin(x/np.pi)+1j*np.cos(x/np.pi))*(1.0-x*x)*(1.0-y*y)
        return ut


    

  

    q0_ini  = initial_conditions(x)
    q1_ini  = np.zeros_like(q0_ini)
    u_ini = np.stack((q0_ini,q1_ini),axis=1)

    print("Running SP_2d Example 1 with scheme ",imex_sch)
    save_dir_case = save_dir
    if not(os.path.exists(save_dir_case)):
          os.makedirs(save_dir_case)
    frm,tt,inv_change_dict,mass_err_l = run_SP_2d_example_1(dt,[x,y],[xix,xiy],kppa,T,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,\
                                                          relax=False,log_errs=False,lap_fac=lap_fac,num_plots=100,p=non_linear_p,data_dir=save_dir_case)
    case_dict={"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"x":x,"xi":xi,\
                 "kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l}
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
          pickle.dump(case_dict,f)
  

    # print("Running Relaxed SP_2d Example 1 with scheme ",imex_sch)
    # save_dir_case = save_dir+"_relaxed"
    # if not(os.path.exists(save_dir_case)):
    #       os.makedirs(save_dir_case)
    # frm,tt,inv_change_dict,mass_err_l = run_SP_2d_example_1(dt,[x,y],[xix,xiy],kppa,T,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,\
    #                                                       relax=2,log_errs=False,lap_fac=lap_fac,num_plots=100,p=non_linear_p,data_dir=save_dir_case)
    # case_dict={"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"x":x,"xi":xi,\
    #              "kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l}
    # with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
    #       pickle.dump(case_dict,f)
    


    
    
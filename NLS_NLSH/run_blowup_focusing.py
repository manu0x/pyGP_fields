import os
import sys
import pickle
import jax
import jax.numpy as jnp
import numpy as np

cwd = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from NLS_functions_p_nonlinearity import *





if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

   

   
    imex_sch = str(sys.argv[1])
    A_im,A_ex,C,b_im,b_ex,imex_stages = choose_imex(imex_sch)  

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex)


    # List of tau values for which u wanna run simulation
    
    
    frame_dict_list = []
    frame_dict_relax_list = []
    frame_dictNLS_list = []
    frame_dictNLS_relax_list = []

    t_list = []

    kppa = 1.0
    non_linear_p = 5.0
    
    dt=1e-6     ## Choose dt
    t_ini = 0.0
    T = 0.004
    

    m = 1600000  ## Number of grid points
    xL = -8.0; xR = 8.0; L = xR-xL
    x = np.arange(-m/2,m/2)*(L/m)
    xj = jnp.array(x)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    tau_list = [0.001,0.00001]

    inv_list = [H,I1,I2]

    
    case="imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    save_dir = "./data/"+"blowup_focusing"
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    save_dir = save_dir+"/"+case
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    
    
    
    
    

    def initial_conditions(t,x,q):
        #N = len(x)
        ut = 5.0*jnp.exp(-10.0*jnp.square(x))
        return ut

    def initial_conditions_real(t,x,q):
        return jnp.real(initial_conditions(t,x,q))    
    def initial_conditions_imag(t,x,q):
        return jnp.imag(initial_conditions(t,x,q))    
    
    ini_cond_real_x =  grad(initial_conditions_real,1)
    ini_cond_imag_x =  grad(initial_conditions_imag,1)
  
    ini_cond_real_x_vm =  jax.vmap(grad(ini_cond_real_x,1),(0,0,None))
    ini_cond_imag_x_vm =  jax.vmap(grad(ini_cond_imag_x,1),(0,0,None))


    

    sol_real = initial_conditions_real(t_ini*np.ones_like(x),xj,kppa)
    sol_imag = initial_conditions_imag(t_ini*np.ones_like(x),xj,kppa)
    sol = np.array(sol_real)+1j*np.array(sol_imag)

    dx_sol_real=ini_cond_real_x_vm(t_ini*np.ones_like(x),xj,kppa)
    dx_sol_imag=ini_cond_imag_x_vm(t_ini*np.ones_like(x),xj,kppa)
    dx_sol = np.array(dx_sol_real)+1j*np.array(dx_sol_imag)

    

  

    q0_ini  = sol
    q1_ini  = dx_sol
    u_ini = np.stack((q0_ini,q1_ini),axis=1)

    print("Running NLS")
    save_dir_case = save_dir+"/NLS"
    if not(os.path.exists(save_dir_case)):
         os.makedirs(save_dir_case)
    frm,tt,inv_change_dict,mass_err_l = run_nls_example(dt,x,xi,kppa,T,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,\
                                                         relax=False,log_errs=False,num_plots=200,p=non_linear_p,data_dir=save_dir_case)
    case_dict={"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"x":x,"xi":xi,\
                "kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l}
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
         pickle.dump(case_dict,f)
  

    print("Running Relaxed NLS")
    save_dir_case = save_dir+"/relaxed_NLS"
    if not(os.path.exists(save_dir_case)):
         os.makedirs(save_dir_case)
    frm,tt,inv_change_dict,mass_err_l = run_nls_example(dt,x,xi,kppa,T,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,\
                                                         relax=True,log_errs=False,num_plots=200,p=non_linear_p,data_dir=save_dir_case)
    case_dict={"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"x":x,"xi":xi,\
                "kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l}
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
         pickle.dump(case_dict,f)
    

    for tau in tau_list:

        ## Options petviashvili,cubic
        ###  LOAD Initial conditions and grid setup i.e. x,xi,m,L,T,kppa,etc.
        
       
            
       

        
        
        
        lmda_list = setup_tau(imx,dt,xi,tau)

        print("Running  Hyperbolized NLS with tau=",tau)
        save_dir_case = save_dir+"/NLSH_tau_"+str(tau)
        if not(os.path.exists(save_dir_case)):
            os.makedirs(save_dir_case)
        frm,tt,inv_change_dict,mass_err_l  = run_nls_hyper_example(dt,x,xi,kppa,T,tau,lmda_list,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,\
                                                                                     relax=False,log_errs=False,num_plots=200,p=non_linear_p,data_dir=save_dir_case)
        case_dict={"tau":tau,"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,\
                   "x":x,"xi":xi,"kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l}
        with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
            pickle.dump(case_dict,f)

    
        print("Running Relaxed Hyperbolized NLS with tau=",tau)
        save_dir_case = save_dir+"/relaxed_NLSH_tau_"+str(tau)
        if not(os.path.exists(save_dir_case)):
            os.makedirs(save_dir_case)
        frm,tt,inv_change_dict,mass_err_l = run_nls_hyper_example(dt,x,xi,kppa,T,tau,lmda_list,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,\
                                                                                     relax=True,log_errs=False,num_plots=200,p=non_linear_p,data_dir=save_dir_case)
        case_dict={"tau":tau,"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,\
                   "x":x,"xi":xi,"kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_l}
        with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
            pickle.dump(case_dict,f)



    
    #file_name="tets.pkl"
    # Save lists of dicts containing lists of frames & corresponding times for each tau in a file
   
    
    
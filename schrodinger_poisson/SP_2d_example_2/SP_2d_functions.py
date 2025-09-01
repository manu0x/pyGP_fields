import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import math

from jax import grad
import pickle
import os

from scipy import integrate
import sympy as sp
import numpy as np
import sys

cwd = "/".join(os.getcwd().split("/")[:-2])
sys.path.append(cwd)
sys.path.append(cwd+"/time_integrators")


from GPE import GPE_scalar_field_1d2c
from GPE import GPE_scalar_field_1d2c_relax
from GPE import GPE_scalar_field_multirelax


fft = np.fft.fftn
ifft = np.fft.ifftn




##############################  CHoosing ImEX Scheme  

#Choose ImEx scheme

def choose_imex(imex_scheme="default"):
    from Biswas_Ketcheson_TimeIntegrators import ImEx_schemes
    from Biswas_Ketcheson_TimeIntegrators import load_imex_scheme

    if imex_scheme=="default":
        #print("Choosin default ImEx")
        A_ex    = np.array([[0,0,0],[5/6.,0,0],[11/24,11/24,0]])
        A_im = np.array([[2./11,0,0],[205/462.,2./11,0],[2033/4620,21/110,2/11]])
        b_ex = np.array([24/55.,1./5,4./11])
        b_im = b_ex     
        C=None
        imex_stages=3
    if imex_scheme=="a" or imex_scheme=="ImEx3" :
        #3rd order ImEx with b This method is taken from Implicit-explicit 
        # Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.
        #print("Using 3rd order ImEx with b  This method is taken from Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.")

        A_im,A_ex,C,b_im,b_hat = ImEx_schemes(4,3,2,2)
        b_ex = b_im
        imex_stages=4

    if imex_scheme=="b" or imex_scheme=="ARK3(2)4L[2]SA":
        #3rd order ImEx with b. This method is taken from Additive Runge–Kutta schemes 
        #for convection–diffusion–reaction equations by Kennedy and Carpenter.
        #print("Using 3rd order ImEx with b . This method is taken from Additive Runge–Kutta schemes for convection–diffusion–reaction equations by Kennedy and Carpenter.")
        A_im,A_ex,C,b_im,b_hat = ImEx_schemes(4,3,2,3)
        b_ex = b_im
        imex_stages=4
    if imex_scheme=="c" or imex_scheme=="ImEx4" or imex_scheme=="ARK4(3)6L[2]SA" :
        #4th order ImEx with b and 3rd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
        #for convection–diffusion–reaction equations by Kennedy and Carpenter.
        #print("4th order ImEx with b. This method is taken from Additive Runge–Kutta scheme for convection–diffusion–reaction equations by Kennedy and Carpenter.")
        A_im,A_ex,C,b_im,b_hat = ImEx_schemes(6,4,3,4)
        b_ex = b_im
        imex_stages=6

    if imex_scheme=="SSP2-ImEx(3,3,2)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    if imex_scheme=="SSP3-ImEx(3,4,3)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    if imex_scheme=="AGSA(3,4,2)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    if imex_scheme=="ARS(4,4,3)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    return A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages





################################################################



def rhs_linear(uft,u,t,xi2,kppa,lap_fac):
    
        
    #rho = u[:m]; q = u[m:];
        q0 = u
        v = np.zeros_like(u)
        q0hat = uft
     
        q0_x = ifft(-q0hat*(xi2),u.shape)
  
        rhs_q0 = lap_fac*1j*q0_x
    
        v = rhs_q0*t
   
        return v












######################   Define Norms  ###################

def Lp_norm(f,p):
    abs_f = np.abs(f)
  
    if p<1:
        print("L^p Error, p should be >=1")
        norm = None

    elif p==1:

        norm = np.mean(abs_f)
    elif p==np.inf:
        norm = np.max(abs_f)
    else:
        p = float(p)
        norm = np.mean(np.power(abs_f,p))
        norm = np.power(norm,1.0/p)
    return  norm




## Conserved Quantities   ##############



def mass(u,t,xi2,kppa,lap_fac):
        
        
    
        q0 = u
        q0sqr = np.square(np.abs(q0))
   

        v = np.sum(np.conj(q0)*q0)
    

        return np.abs(v)#-0.5*np.sum(q0sqr+tau*q1sqr)


def energy(u,t,xi2,kppa,lap_fac):
        u_ft = fft(u,u.shape)
        
        V_ft = -fft(np.square(np.abs(u)),u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft[0] = 0.0 
        
        E_p = np.sum(xi2*np.abs(V_ft)**2)*0.5  ## Potential energy

        
        E_k = np.sum(xi2*np.abs(u_ft)**2)*0.5  ## Kinetic energy

        return np.abs(lap_fac*t*E_k - 0.5*kppa*np.sqrt(t)*E_p)
        

   









############### SP-2d euqtaion   ################################################

def run_SP_2d_example_1(dt,X,Xi,kppa,T,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,relax=False,log_errs=False,lap_fac=1.0,num_plots=100,p=3.0,
                    data_dir=None):
    
    dim_n = len(X)
    x=X[0]
    y=X[1]

    xi = Xi[0]
    yi = Xi[1]
    
    xi2 = np.zeros_like(Xi[0])
    for i in range(len(Xi)):
        xi2 = xi2+np.square(Xi[i])

    def rhs_nonlinear(u,uft,t,xi2,kppa,lap_fac):
    #Evaluate the nonlinear term
       
          
        V_ft = -fft(np.square(np.abs(u)),u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft[0] = 0.0  ## Set mean value of potential to zero
        V = (ifft(V_ft,V_ft.shape)).real
        q_rhs = -1j*kppa*V*u*np.sqrt(t)
       
      
        return q_rhs




    
    tmax = T

    

    
    nplt = np.floor((tmax/num_plots)/dt)
    nmax = int(round(tmax/dt))
    #print(nplt,"nmax",nmax)
    
    m = x.shape[0] 


    
   
    tt = []
    err_l = []
    mass_l = []
    mass_err_l=[]

    

    lmbda = lap_fac*xi2
    
    #rhoq = GPE_scalar_field_1d2c(m,2,rhs_linear,rhs_nonlinear,imx,u_ini)
    if relax==2:
        rhoq = GPE_scalar_field_multirelax(2,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0],relax=1,conserve_list=[mass,energy])
    elif relax==1:
        rhoq = GPE_scalar_field_multirelax(2,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0],relax=1,conserve_list=[mass])
    else:
        rhoq = GPE_scalar_field_multirelax(2,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0])
    

    inv_ini = [f(u_ini[:,0],xi2,kppa,lap_fac) for f in inv_list]
    #print("Ini invariant", inv_ini)
    
    n=0
    print_cntr=0
    t=0.0
    frames = [u_ini[:,0],]
    tt.append(t)
   
    while (t<=tmax):
        
        #print(n)
        for k in range(imx.s):
            rhoq.update_stage_sum(k,dt)
            im_t = t+rhoq.im_C[k]*dt
            rhoq.do_fft(k,lmbda*im_t,dt)

            rhoq.update_K(k,dt,t,xi2,kppa,lap_fac)
           
            
            
            
        rhoq.sum_contributions(dt,t,xi2,kppa,lap_fac)
        
        
        if relax:
            t = t+(1.0+np.sum(rhoq.rel_gamma))*dt
        else:
            t = t+dt
        
        u = rhoq.psi
        if math.isnan(np.mean(np.abs(u))+t) or math.isinf(np.mean(np.abs(u))+t):
                print("NaN detected at time ",t,"at time step",n)
                if data_dir is not None:
                    np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
                return frames,tt,inv_change_dic,mass_err_l
        # elif relax:
        #     if np.min(rhoq.rel_gamma)<(1e-10):
        #         print("Relaxation parameter too small at time ",t,"at time step",n)
        #         if data_dir is not None:
        #             np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
        #         return frames,tt,inv_change_dic,mass_err_l
        
        if (np.mod(n,nplt)) == 0 :
            print("Time ",t," step ",n)
            if data_dir is not None:
                np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
            else:
                frames.append(rhoq.psi)
            
            
            u = rhoq.psi
            print_cntr+=1
            
            inv_c = [f(u,t,xi2,kppa,lap_fac) for f in inv_list]

            inv_change_relative = [np.abs((ini-fin)/(ini+1e-12) )for ini,fin in zip(inv_ini,inv_c)]
            inv_change = [np.abs(ini-fin) for ini,fin in zip(inv_ini,inv_c)]
            inv_change_dic = {"change":inv_change,"relative change":inv_change_relative}
            mass_err_l.append(np.abs(inv_ini[1]-inv_c[1]))
            tt.append(t)

            if log_errs==True:
                if exact_soln_np!=None:
                    sol = exact_soln_np(t*np.ones_like(x),x,kppa)
                
                    q0_diff = u-sol
                    
            
                    q0_err_Linf= Lp_norm(q0_diff,np.inf)
                    q0_err_L1 = Lp_norm(q0_diff,1)
                    q0_err_L2 = Lp_norm(q0_diff,2)

                    
                    errs = [q0_err_L1,q0_err_L2,q0_err_Linf]
                else:
                    errs = [None,None,None]
                err_l.append(errs)
                
            #err = sol_err(psi.psi,t,nx,T,max_err)
           # err_l.append(err)
            
           # mass = psi_1.calc_mass()+psi_2.calc_mass()
           # mass_err = (mass-mass_ini)/mass_ini
            
          ##  mass_l.append(mass)
           # mass_err_l.append(mass_err)
            
            #print("time ",t/tmax,t)
        n=n+1

    u = rhoq.psi
    frames.append(u)
    tt.append(t)
    if log_errs==True:
        if exact_soln_np!=None:
            sol = exact_soln_np(t*np.ones_like(x),x,kppa)
            
            q0_diff = u-sol
        
                
            q0_err_Linf= Lp_norm(q0_diff,np.inf)
            q0_err_L1 = Lp_norm(q0_diff,1)
            q0_err_L2 = Lp_norm(q0_diff,2)

            
            errs = [q0_err_L1,q0_err_L2,q0_err_Linf]
        else:
            errs = [None,None,None]

    inv_fin = [f(u,t,xi2,kppa,lap_fac) for f in inv_list]

    inv_change_relative = [np.abs((ini-fin)/(ini+1e-12) )for ini,fin in zip(inv_ini,inv_fin)]
    inv_change = [np.abs(ini-fin) for ini,fin in zip(inv_ini,inv_fin)]
    inv_change_dic = {"change":inv_change,"relative change":inv_change_relative}
    mass_err_l.append(np.abs(inv_ini[1]-inv_fin[1]))
    #err =  sol_err(psi.psi,t,nx,T,max_err)
    #err_l.append(err)
            
   # mass = psi_1.calc_mass()+psi_2.calc_mass()
    #mass_err = (mass-mass_ini)/mass_ini
            
   # mass_l.append(mass)
   # mass_err_l.append(mass_err)
       
    return frames,tt,inv_change_dic,mass_err_l


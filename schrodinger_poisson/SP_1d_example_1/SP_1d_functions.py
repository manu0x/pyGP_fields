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


fft = np.fft.fft
ifft = np.fft.ifft




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



def rhs_linear(uft,u,xi,tau,kppa,lap_fac):
    
        m = len(xi)
    #rho = u[:m]; q = u[m:];
        q0 = u
        v = np.zeros_like(u)
        q0hat = uft
     
        q0_x = ifft(-q0hat*(xi**2))
  
        rhs_q0 = lap_fac*1j*q0_x
    
        v = rhs_q0
   
        return v




def dxi_u(uft,u,xi,i):
    q0hat = uft[:,0]
    q1hat = uft[:,1]

    v = np.zeros_like(u)

    dxi_q0 = ifft(np.power(1j*xi,i)*q0hat)
    dxi_q1 = ifft(np.power(1j*xi,i)*q1hat)

    v[:,0] = dxi_q0; v[:,1] = dxi_q1

    return v

def dxi_u_1c(uft,u,xi,i):
    q0hat = uft

    v = np.zeros_like(u)

    dxi_q0 = ifft(np.power(1j*xi,i)*q0hat)


    v[:] = dxi_q0

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



def mass(u,xi,kppa,lap_fac):
        
        u_ft = fft(u)
    
        q0 = u
        q0sqr = np.square(np.abs(q0))
   

        v = np.sum(np.conj(q0)*q0)
    

        return np.abs(v)#-0.5*np.sum(q0sqr+tau*q1sqr)


def energy(u,xi,kppa,lap_fac):
        u_ft = fft(u)
        
        V_ft = -fft(np.square(np.abs(u)))/(xi*xi+1e-14) ## Poisson eqn. in FT space
        V_ft[0] = 0.0 
        V_x = ifft(1j*xi*V_ft)
        E_p = np.sum(np.conj(V_x)*V_x)

        u_x = ifft(1j*xi*u_ft)
        E_k = np.sum(np.conj(u_x)*u_x)

        
 

        return np.abs(lap_fac*E_k - 0.5*kppa*E_p)
        

   

################################################################################################################
################    Setup Lin alg. opetrations  ################################################################



def setup_tau(imx,dt,xi,tau):
    lmda_list=[]
    chklist=[]
    for s in range(imx.s):
        impf = imx.im_A[s][s]
        #print(s,imx.s,impf,tau)
        omega = xi*dt*impf
        bta = -xi*dt*impf/tau
        chk = np.ones_like(xi)
    
        l = [[[chk[i],omega[i]],[bta[i],(chk[i]-1j*dt*impf/tau)]] for i in range(len(xi))]
        a=chk
        b=omega
        c = bta
        d = (chk-1j*dt*impf/tau)
        det_M = a*d-b*c
        
        l_inv = np.array( [ [[d[i]/det_M[i],-b[i]/det_M[i]],[-c[i]/det_M[i],a[i]/det_M[i]]] for i in range(len(xi))])
        mt_inv = l_inv
    #l = [[[1.0+impf*dt*1j*np.square(xi[i]),0.0 ],[bta[i],(chk[i]-1j*dt)/tau]] for i in range(len(xi))]

        mt = np.array(l)
        #print("shape check",mt.shape,mt_inv.shape)
        #chklist.append(np.matmul(mt,mt_inv))
        lmdamat = mt_inv#np.linalg.inv(mt)

        lmda_list.append(lmdamat)
    #print(mt.shape, mt.dtype,lmdamat.shape,q0_ini.shape,q1_ini.shape,u_ini.shape)

    return lmda_list#,chklist



#####################
##################### Code to run the example 







############### SP-1d euqtaion   ################################################

def run_SP_1d_example_1(dt,x,xi,kppa,T,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,relax=False,log_errs=False,lap_fac=1.0,num_plots=100,p=3.0,
                    data_dir=None):
    
       
    def rhs_nonlinear(u,uft, xi,V,kppa,lap_fac):
    #Evaluate the nonlinear term
        m = len(xi)
          
        V_ft = -fft(np.square(np.abs(u)))/(xi*xi+1e-14) ## Poisson eqn. in FT space
        V_ft[0] = 0.0  ## Set mean value of potential to zero
        V = (ifft(V_ft)).real
        q_rhs = -1j*kppa*V*u
       
      
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

    

    lmbda = lap_fac*xi**2
    
    #rhoq = GPE_scalar_field_1d2c(m,2,rhs_linear,rhs_nonlinear,imx,u_ini)
    if relax==2:
        rhoq = GPE_scalar_field_multirelax(1,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0],relax=1,conserve_list=[mass,energy])
    elif relax==1:
        rhoq = GPE_scalar_field_multirelax(1,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0],relax=1,conserve_list=[mass])
    else:
        rhoq = GPE_scalar_field_multirelax(1,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0])
    

    inv_ini = [f(u_ini[:,0],xi,kppa,lap_fac) for f in inv_list]
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
            
            rhoq.do_fft(k,lmbda,dt)
            
            rhoq.update_K(k,dt,xi,None,kppa,lap_fac)
           
            
            
            
        rhoq.sum_contributions(dt,xi,kppa,lap_fac)
        
        
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
            
            inv_c = [f(u,xi,kppa,lap_fac) for f in inv_list]

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

    inv_fin = [f(u,xi,kppa,lap_fac) for f in inv_list]

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


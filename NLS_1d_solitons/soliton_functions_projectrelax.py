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

cwd = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(cwd)
sys.path.append(cwd+"/time_integrators")


# from GPE import GPE_scalar_field_1d2c
# from GPE import GPE_scalar_field_1d2c_relax
from GPE import GPE_scalar_field_projectrelax_1d


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



def mass(u,xi2,kppa,lap_fac,mu,Q=None):
        
        
    
        q0 = u
  
   

        v = np.mean(np.conj(q0)*q0)
    

        return np.abs(v)#-0.5*np.sum(q0sqr+tau*q1sqr)


def energy(u,t,xi2,kppa,lap_fac,Q=None):
        u_ft = fft(u)
        
        V_ft = -fft(np.square(np.abs(u)))/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft[0] = 0.0 
        
        E_p = np.sum(xi2*np.abs(V_ft)**2)  ## Potential energy

        
        E_k = np.sum(xi2*np.abs(u_ft)**2)  ## Kinetic energy

        return lap_fac*t*E_k - 0.5*kppa*np.sqrt(t)*E_p
        





   
def kin_energy_2(u,xi,kppa,lap_fac,mu,Q=None):
        
        u_ft = fft(u)
        
        xi2 = np.square(xi)
        D2_u = ifft(-u_ft*(xi2))
        E_k2 = -np.mean(np.conj(u)*D2_u).real  ## Kinetic energy




        #return 4.0*lap_fac*lap_fac*E_k   ## If Ek1 is used
        return E_k2  ## If Ek2 is used

def kin_energy_1(u,xi,kppa,lap_fac,mu,Q=None):
        
        u_ft = fft(u)
 
        ##################################################### EP 1 #################################
        u_x = ifft(1j*xi*u_ft)
       
        E_k = np.mean(np.abs(np.conj(u_x)*u_x)) ## Kinetic energy





        return E_k   ## If Ek1 is used

def kin_energy_0(u,xi,kppa,lap_fac,mu,Q=None):
        
        u_ft = fft(u)
 
        ##################################################### EP 0 #################################
        E_k0 = np.mean(xi*xi*np.abs(u_ft)**2)




        return E_k0   ## If EP0 is used
   

def pot_energy(u,xi,kppa,lap_fac,mu,Q=None):
    
        
        return 0.5*kppa*np.mean(np.power(np.abs(u),4.0))   

        






def Q_dummy(u,xi2,kppa,lap_fac,mu,Q):
     
     return Q


############################################################hjjdhkckdjhcdjhdjkcd
############### SP-2d euqtaion   ################################################

def run_SP_2d_example_1(dt,X,Xi,kppa,t_ini,T,L,imx,inv_list,u_ini,kin_type=None,exact_soln_np=None,dx_soln_jx=None,relax=False,log_errs=False,lap_fac=1.0,num_plots=100,p=3.0,
                    data_dir=None):
    
    if kin_type=="Ek1":
        kin_energy = kin_energy_1
    elif kin_type=="Ek2":
        kin_energy = kin_energy_2

    kin_list = [kin_energy_0,kin_energy_1,kin_energy_2]
    print("Using kinetic energy type ",kin_type)
    dim_n = len(X)
    x=X[0]
    

    xi = Xi[0]

    
    xi2 = np.zeros_like(Xi[0])
    for i in range(len(Xi)):
        xi2 = xi2+np.square(Xi[i])


    def rhs_linear(uft,u,t,xi,kppa,lap_fac,mu):
    
        xi2 = np.square(xi)

        
     
        q0_x = ifft(-uft*(xi2))
  
        rhs_q0 = lap_fac*1j*q0_x
    

        return rhs_q0

    def rhs_nonlinear(u,uft,t,xi,kppa,lap_fac,mu):
    #Evaluate the nonlinear term
      
        q_rhs = 1j*kppa*np.square(np.abs(u))*u
       
      
        return q_rhs



    def func2optimize_SP(rel_gamma,u,proj_u,inv_list_old,dt,t,*args):
               # print(rel_gamma.shape,terms.shape,np.dot(rel_gamma,terms).shape)
                u_gamma = u*(1.0-rel_gamma) + rel_gamma*proj_u#np.einsum("i,ij->j", rel_gamma, terms)

                mass_new = np.mean(np.abs(u_gamma)**2)

                proj_u_gamma = np.sqrt(inv_list_old[0]/mass_new)*u_gamma
                #print(inv_list_old[-1])
                
                if relax>1:

                    

                    Q_old = inv_list_old[1]-inv_list_old[2]

                    Ek_new = kin_energy(proj_u_gamma,*args)
                    Ev_new = pot_energy(proj_u_gamma,*args)
                    Q_new = (Ek_new)-(Ev_new)
                    #print(p,q,p_new,q_new,Q_old,Q_new)

                    inv_func = np.array(Q_new-Q_old)
                    #inv_func = np.array([mass_new-inv_list_old[0],p*(Ek_new-inv_list_old[1])-q*(Ev_new-inv_list_old[2])])
                else:
                    inv_func = mass_new-inv_list_old[0]
                return inv_func
    
    
            

    
    tmax = T

    

    
    nplt = np.floor((tmax/num_plots)/dt)
    nmax = int(round(tmax/dt))
    #print(nplt,"nmax",nmax)
    
    m = x.shape[0] 


    
   
    tt = []
    err_l = []
    mass_l = []
    mass_err_l=[]
    energy_err_l = []
    eq_Fd_list = []
    E_l = []

    

    lmbda = lap_fac*xi2
    
    #rhoq = GPE_scalar_field_1d2c(m,2,rhs_linear,rhs_nonlinear,imx,u_ini)
    if relax==2:
        rhoq = GPE_scalar_field_projectrelax_1d(1,m,rhs_linear,rhs_nonlinear,imx,u_ini,relax=2,conserve_list=[mass,kin_energy,pot_energy,Q_dummy],func2optimize=func2optimize_SP)
    elif relax==1:
        rhoq = GPE_scalar_field_projectrelax_1d(1,m,rhs_linear,rhs_nonlinear,imx,u_ini,relax=1,conserve_list=[mass],func2optimize=func2optimize_SP)
    else:
        rhoq = GPE_scalar_field_projectrelax_1d(1,m,rhs_linear,rhs_nonlinear,imx,u_ini)

    
    #print("Ini invariant", inv_ini)
    print_list = [0.023,0.033,0.088,0.00001,0.00002]
    compare_print_cntr=0

    n=0
    print_cntr=0

    t=t_ini
    frames = [u_ini,]
    tt.append(t)

    mass_ini = np.mean(np.abs(u_ini)**2)
    print("Initial mass is",mass_ini,L)
    mu = mass_ini#*L*L
    ke_last = [kinfunc(rhoq.psi,xi,kppa,lap_fac,mu) for kinfunc in kin_list]
    pe_last = pot_energy(rhoq.psi,xi,kppa,lap_fac,mu)

    np.savez(data_dir+"/frame_initial",frame=rhoq.psi)

    print("B sum",np.sum(rhoq.ex_B),"mu",mu)
    
    while (t<=tmax):

        
   
        
        #print(n)
        Q_int = 0.0
        for k in range(imx.s):
            rhoq.update_stage_sum(k,dt)

           

            
            rhoq.do_fft(k,lmbda,dt)

           

            rhoq.update_K(k,dt,t,xi,kppa,lap_fac,mu)
           


        #print("Q_int:", Q_int)
        rhoq.sum_contributions(dt,t,xi,kppa,lap_fac,mu,Q_int)
        
        
        if relax:
            t_m = t+0.5*rhoq.rel_gamma*dt
            t = t+rhoq.rel_gamma*dt
            
        else:
            t_m = t+0.5*dt
            t = t+dt
        # p = 1.0#np.power(t_m,-1.5)
        # q = t#1.0/np.sqrt(t_m)
        
        # p_m = 1.0
        # q_m = t_m
        
        u = rhoq.psi
        ke = [kinfunc(u,xi,kppa,lap_fac,mu) for kinfunc in kin_list]
        pe = pot_energy(u,xi,kppa,lap_fac,mu)

        eq_E_new = [k-pe for k in ke]
        eq_E_last = [k-pe_last for k in ke_last]

        eq_E_l = [k-l for k,l in zip(eq_E_new,eq_E_last)]
        
        eq_E_fd = 0.0
        pe_last = pe
        ke_last = ke

        if math.isnan(np.mean(np.abs(u))+t) or math.isinf(np.mean(np.abs(u))+t) :
                if print_cntr<=39:
                    print("NaN detected at time ",t,"at time step",n,rhoq.rel_gamma)
                else:
                     print("Too many steps",print_cntr)
                if data_dir is not None:
                    np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
                return frames,tt,inv_change_dic,mass_l,mass_err_l,energy_err_l,eq_Fd_list
        # elif relax:
        #     if np.min(rhoq.rel_gamma)<(1e-10):
        #         print("Relaxation parameter too small at time ",t,"at time step",n)
        #         if data_dir is not None:
        #             np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
        #         return frames,tt,inv_change_dic,mass_err_l

        if (t>(print_list[compare_print_cntr]-0.5*dt) and t<(print_list[compare_print_cntr]+0.5*dt)  ):
             np.savez(data_dir+"/fm_"+str(t)[:6],frame=rhoq.psi)
             print("###############################################")
             print("Writing data at",print_list[compare_print_cntr])
             print("###############################################")
             compare_print_cntr=compare_print_cntr+1
             
        #print("Time ",t," step ","Gamma",n,np.sum(rhoq.rel_gamma),rhoq.rel_gamma)
        if (np.mod(n,nplt)) == 0 :
            E_l.append(eq_E_new)
            #print("Time ",t," step ",n,rhoq.rel_gamma,np.sum(rhoq.rel_gamma))
            if relax:
                 print("Time ",t," step ","Gamma sum",n,rhoq.rel_gamma)
            else:
                 print("Time ",t," step ",n,)#1.0+np.sum(rhoq.rel_gamma),rhoq.rel_gamma)
            u = rhoq.psi
            print_cntr+=1
            if data_dir is not None:
                np.savez(data_dir+"/frame_"+str(t)[:6],frame=rhoq.psi)
            else:
                frames.append(rhoq.psi)
            
            
            

            mass_now = np.mean(np.abs(u)**2)
            mass_change = np.abs(mass_now-mass_ini)
            mass_relative_change = mass_change/(mass_ini+1e-12)
            
            
            mass_l.append(mass_now)
            energy_violation = [np.abs(eq_E) for eq_E in eq_E_l]
            mass_err_l.append(mass_relative_change)
            energy_err_l.append(energy_violation)
            eq_Fd_list.append(np.abs(eq_E_fd))
            tt.append(t)

            #print("Mass",mass_now,mass_relative_change,energy_violation)

            inv_change_dic = {"mass change":mass_change,"relative mass change":mass_relative_change,"energy violation":energy_violation}
            print("Invariants change:",energy_violation,mass_relative_change)
            if log_errs==True:
                if exact_soln_np!=None:
                    sol = exact_soln_np(t,x,kppa)
                
                    q0_diff = u-sol
                    
            
                    q0_err_Linf= Lp_norm(q0_diff,np.inf)
                    q0_err_L1 = Lp_norm(q0_diff,1)
                    q0_err_L2 = Lp_norm(q0_diff,2)

                    
                    errs = [q0_err_L1,q0_err_L2,q0_err_Linf]
                    #print("Errors L1,L2,Linfty:",errs)
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
    

    mass_now = np.mean(np.abs(u)**2)
    mass_change = np.abs(mass_now-mass_ini)
    mass_relative_change = mass_change/(mass_ini+1e-12)

    ke = [kinfunc(u,xi,kppa,lap_fac,mu) for kinfunc in kin_list]
    pe = pot_energy(u,xi,kppa,lap_fac,mu)

    eq_E_new = [k-pe for k in ke]
    eq_E_last = [k-pe_last for k in ke_last]

    eq_E_l = [k-l for k,l in zip(eq_E_new,eq_E_last)]
    E_l.append(eq_E_new)
            
    energy_violation = [np.abs(eq_E) for eq_E in eq_E_l]
    mass_err_l.append(mass_relative_change)
    energy_err_l.append(energy_violation)

    inv_change_dic = {"mass change":mass_change,"relative mass change":mass_relative_change,"energy violation":energy_violation}
    #err =  sol_err(psi.psi,t,nx,T,max_err)
    #err_l.append(err)
            
   # mass = psi_1.calc_mass()+psi_2.calc_mass()
    #mass_err = (mass-mass_ini)/mass_ini
            
   # mass_l.append(mass)
   # mass_err_l.append(mass_err)
       
    return frames,tt,inv_change_dic,mass_l,mass_err_l,energy_err_l,E_l,err_l


















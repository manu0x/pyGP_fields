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



#from GPE import GPE_scalar_field_1d2c
#from GPE import GPE_scalar_field_1d2c_relax
from GPE import GPE_scalar_field_multirelax_test


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



def mass(u,xi,kppa,lap_fac,mu):
        
        
    
        q0 = u
        

        v = np.mean(np.conj(q0)*q0)
    

        return np.abs(v)#-0.5*np.sum(q0sqr+tau*q1sqr)


def energy(u,t,xi2,kppa,lap_fac):
        u_ft = fft(u,u.shape)
        
        V_ft = -fft(np.square(np.abs(u)),u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft[0] = 0.0 
        
        E_p = np.sum(xi2*np.abs(V_ft)**2)  ## Potential energy

        
        E_k = np.sum(xi2*np.abs(u_ft)**2)  ## Kinetic energy

        return lap_fac*t*E_k - 0.5*kppa*np.sqrt(t)*E_p
        


   
def kin_energy(u,xi,kppa,lap_fac,mu,Q=None):
        xix = xi[0]
        xiy = xi[1]
        u_ft = fft(u,u.shape)
        
        u_x = ifft(1j*xix*u_ft,u.shape)
        u_y = ifft(1j*xiy*u_ft,u.shape)

        
        
      

        
        E_k = np.mean(np.abs(np.conj(u_x)*u_x + np.conj(u_y)*u_y)) ## Kinetic energy

        return 4.0*lap_fac*lap_fac*E_k 

def pot_energy(u,xi,kppa,lap_fac,mu,Q=None):
        xix = xi[0]
        xiy = xi[1]
        xi2 = xix*xix + xiy*xiy
        V_ft = -fft(np.square(np.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft[0,0] = 0.0 

        V_x = ifft(1j*xix*V_ft,u.shape)
        V_y = ifft(1j*xiy*V_ft,u.shape)
        
        E_p1 = np.mean(np.abs(np.conj(V_x)*V_x + np.conj(V_y)*V_y)) ## Potential energy

        # V = ifft(V_ft,u.shape).real
        # E_p2 = -np.mean(V*(np.square(np.abs(u))-mu))

        
 
        
        #return 2.0*kppa*lap_fac*(2.0*E_p2-E_p1)
        return 2.0*kppa*lap_fac*E_p1






############### SP-2d euqtaion   ################################################

def run_SP_2d_example_1(dt,X,Xi,kppa,t_ini,T,L,imx,inv_list,u_ini,exact_soln_np=None,dx_soln_jx=None,relax=False,log_errs=False,lap_fac=1.0,num_plots=100,p=3.0,
                    data_dir=None):
    
    dim_n = len(X)
    x=X[0]
    y=X[1]

    xi = Xi[0]
    yi = Xi[1]
    
    xi2 = np.zeros_like(Xi[0])
    for i in range(len(Xi)):
        xi2 = xi2+np.square(Xi[i])

    def rhs_nonlinear(u,uft,t,xi,kppa,lap_fac,mu):
    #Evaluate the nonlinear term
        xix = xi[0]
        xiy = xi[1]
        xi2 = xix*xix + xiy*xiy
        
        V_ft = -fft(np.square(np.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft[0,0] = 0.0  ## Set mean value of potential to zero
        V = (ifft(V_ft,V_ft.shape)).real
        q_rhs = -1j*kppa*V*u/np.sqrt(t)
       
      
        return q_rhs

    def rhs_linear(uft,u,t,xi,kppa,lap_fac,mu):
    
        xix = xi[0]
        xiy = xi[1]
        xi2 = xix*xix + xiy*xiy
   
        v = np.zeros_like(u)
        q0hat = uft
     
        q0_x = ifft(-q0hat*(xi2),u.shape)
  
        rhs_q0 = lap_fac*1j*q0_x
    
        v = rhs_q0/np.power(t,1.5)
   
        return v



    def func2optimize_SP(rel_gamma,u,terms,inv_list_old,dt,t,*args):
               # print(rel_gamma.shape,terms.shape,np.dot(rel_gamma,terms).shape)
                u_gamma = u + np.einsum("i,ijk->jk", rel_gamma, terms)
                mass_new = np.mean(np.abs(u_gamma)**2)
                
                
                if relax>1:
                    t_half = t+0.5*(1.0+np.sum(rel_gamma))*dt
                    dt_now = (1.0+np.sum(rel_gamma))*dt
                    #if t<3e-3:
                    #    print("Time updated to ", t_half,rel_gamma)
                    #print("t_half",t_half,np.sum(rel_gamma))
                    p = 1.0#/(np.sqrt(t_half)*t_half)#(1.0/np.power(t,1.5))
                    q = t_half#(1.0/np.sqrt(t_half))
                    #print(p,q)
                    
                    Ek_new = kin_energy(u_gamma,*args)
                    Ev_new = pot_energy(u_gamma,*args)

                    inv_func = np.array([(mass_new-inv_list_old[0]),p*(Ek_new-inv_list_old[1])- q*(Ev_new-inv_list_old[2])])
                    #print(inv_func)
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
    int_energy_err_l =[]

    

    lmbda = lap_fac*xi2
    
    #rhoq = GPE_scalar_field_1d2c(m,2,rhs_linear,rhs_nonlinear,imx,u_ini)
    if relax==2:
        rhoq = GPE_scalar_field_multirelax_test(2,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0],relax=2,conserve_list=[mass,kin_energy,pot_energy],func2optimize=func2optimize_SP)
    elif relax==1:
        rhoq = GPE_scalar_field_multirelax_test(2,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0],relax=1,conserve_list=[mass],func2optimize=func2optimize_SP)
    else:
        rhoq = GPE_scalar_field_multirelax_test(2,m,rhs_linear,rhs_nonlinear,imx,u_ini[:,0])
    

    
    #print("Ini invariant", inv_ini)
    print_list = [0.023,0.033,0.088,0.00001,0.00002]
    compare_print_cntr=0
    
    n=0
    print_cntr=0
    t=t_ini
    frames = [u_ini[:,0],]
    tt.append(t)
    
    mass_ini = np.mean(np.abs(u_ini[:,0])**2)
    mu = mass_ini#*L*L
    print("initial mass is",mass_ini)
    ke_last = kin_energy(rhoq.psi,Xi,kppa,lap_fac,mu)
    pe_last = pot_energy(rhoq.psi,Xi,kppa,lap_fac,mu)

    np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
   
    while (t<=tmax):
        p_last = 1.0
        q_last = t
        #print(n)
        Q_int=0.0
        for k in range(imx.s):
            rhoq.update_stage_sum(k,dt)
            im_t = t+rhoq.im_C[k]*dt
            lap_t = np.power(im_t,-1.5)
            rhoq.do_fft(k,lmbda*lap_t,dt)

            E_k = kin_energy(rhoq.f,Xi,kppa,lap_fac,mu)
            E_v = pot_energy(rhoq.f,Xi,kppa,lap_fac,mu)

            p_t = 0.0#-1.0/np.power(im_t,2.5)
            q_t = 1.0#-0.5/np.power(im_t,1.5)
            
            Q_int+= rhoq.ex_B[k]*(p_t*(E_k)-q_t*(E_v))*dt

            rhoq.update_K(k,dt,t,Xi,kppa,lap_fac,mu)
           
            
            
            
        rhoq.sum_contributions(dt,t,Xi,kppa,lap_fac,mu)
        
        
        if relax:
            t_m = t+0.5*(1.0+np.sum(rhoq.rel_gamma))*dt
            t = t+(1.0+np.sum(rhoq.rel_gamma))*dt
            
        else:
            t_m = t+0.5*dt
            t = t+dt
        pm = 1.0
        qm = t_m

        p = 1.0
        q = t
        
        u = rhoq.psi
        ke = kin_energy(u,Xi,kppa,lap_fac,mu)
        pe = pot_energy(u,Xi,kppa,lap_fac,mu)

        eq_E = pm*(ke-ke_last)-qm*(pe-pe_last)
        #############
        int_eq_E_new = p*(ke)-q*(pe)
        int_eq_E_last = p_last*(ke_last)-q_last*(pe_last)

        int_eq_E = int_eq_E_new-int_eq_E_last-Q_int
        ###############
        if relax:
             eq_E = eq_E/((1.0+np.sum(rhoq.rel_gamma))*dt)
        else:
             eq_E = eq_E/dt
        ke_last = ke
        pe_last = pe
        

        if math.isnan(np.mean(np.abs(u))+t) or math.isinf(np.mean(np.abs(u))+t):
                print("NaN detected at time ",t,"at time step",n)
                if data_dir is not None:
                    np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
                return frames,tt,inv_change_dic,mass_l,mass_err_l,energy_err_l,int_energy_err_l
        elif relax:
            if (1.0+np.sum(rhoq.rel_gamma))<1e-5:
                print("Gamma upfdate too small",(1.0+np.sum(rhoq.rel_gamma)))
                if data_dir is not None:
                    np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
                return frames,tt,inv_change_dic,mass_l,mass_err_l,energy_err_l,int_energy_err_l
        if print_cntr>500:
            print("Too many stpes",print_cntr)
            if data_dir is not None:
                    np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
            return frames,tt,inv_change_dic,mass_l,mass_err_l,energy_err_l,int_energy_err_l
        
        # elif relax:
        #     if np.min(rhoq.rel_gamma)<(1e-10):
        #         print("Relaxation parameter too small at time ",t,"at time step",n)
        #         if data_dir is not None:
        #             np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
        #         return frames,tt,inv_change_dic,mass_err_l

        if (t>(print_list[compare_print_cntr]-0.5*dt) and t<(print_list[compare_print_cntr]+0.5*dt)  ):
             np.savez(data_dir+"/frame_"+str(print_list[compare_print_cntr]),frame=rhoq.psi)
             print("###############################################")
             print("Writing data at",print_list[compare_print_cntr])
             print("###############################################")
             compare_print_cntr=compare_print_cntr+1
        
        if (np.mod(n,nplt)) == 0 :
            if relax:
                 print("Time ",t," step ",n,1.0+np.sum(rhoq.rel_gamma),rhoq.rel_gamma)
            else:
                print("Time ",t," step ",n)
            u = rhoq.psi
            print_cntr+=1
            if data_dir is not None:
                np.savez(data_dir+"/frame_"+str(print_cntr),frame=rhoq.psi)
            else:
                frames.append(rhoq.psi)
            
            
            

            mass_now = np.mean(np.abs(u)**2)
            mass_change = np.abs(mass_now-mass_ini)
            mass_relative_change = mass_change/(mass_ini+1e-12)
            
            
            mass_l.append(mass_now)
            energy_violation = np.abs(eq_E)
            mass_err_l.append(mass_relative_change)
            energy_err_l.append(energy_violation)
            int_energy_err_l.append(int_eq_E)

            print("Mass",mass_now,mass_relative_change,energy_violation)
            tt.append(t)

            inv_change_dic = {"mass change":mass_change,"relative mass change":mass_relative_change,"energy violation":energy_violation}

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
    

    mass_now = np.mean(np.abs(u)**2)
    mass_change = np.abs(mass_now-mass_ini)
    mass_relative_change = mass_change/(mass_ini+1e-12)
            
    energy_violation = np.abs(eq_E)
    mass_err_l.append(mass_relative_change)
    energy_err_l.append(energy_violation)

    inv_change_dic = {"mass change":mass_change,"relative mass change":mass_relative_change,"energy violation":energy_violation}
    #err =  sol_err(psi.psi,t,nx,T,max_err)
    #err_l.append(err)
            
   # mass = psi_1.calc_mass()+psi_2.calc_mass()
    #mass_err = (mass-mass_ini)/mass_ini
            
   # mass_l.append(mass)
   # mass_err_l.append(mass_err)
       
    return frames,tt,inv_change_dic,mass_l,mass_err_l,energy_err_l,int_energy_err_l


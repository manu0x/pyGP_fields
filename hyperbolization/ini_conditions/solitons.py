import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

import numpy as np
import sys




if __name__=="__main__":
    n_solitons = int(sys.argv[1])
    print("No of solitons in ini. cond. is",n_solitons)
    
    if n_solitons==2:
        q = 8; sol = 2; inv = 1
    if n_solitons==3:
        q = 18; sol = 3; inv = 1
    kppa = q

    if q == 8 and sol == 2 and inv == 1: 
    
        print("2 soliton selected")
        xL = -16; xR = 16; L = xR-xL; m1 = 16; N =m1*L; t0 = 0; DT = [0.01,0.01]; SP_DT = [0.01,0.01]; T = 5
    elif q == 18 and sol == 3 and inv == 1: 
        
        print("3 soliton selected")
        xL = -8; xR = 8; L = xR-xL; m1 = 32; N =m1*L; t0 = 0; DT = [0.01,0.01]; SP_DT = [0.01,0.01] ; T = 5

    m = 1024
    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    def NLS_True_Sol(t,x,q):
        #N = len(x)
        if q == 2:
            ut = jnp.exp(1j*t)/jnp.cosh(x)
        elif q == 8:
            sechx = 1./jnp.cosh(x)
            ut = jnp.exp(1j*t)*sechx*( 1+(3/4)*sechx**2*(jnp.exp(8*1j*t)-1) )/( 1-(3/4)*sechx**4*jnp.sin(4*t)**2 )
        elif q == 18:
            ut = (2*(3*jnp.exp(t*25*1j)*jnp.exp(x) + 15*jnp.exp(t*9*1j)*jnp.exp(9*x) + 48*jnp.exp(t*25*1j)*jnp.exp(7*x) + 48*jnp.exp(t*25*1j)*jnp.exp(11*x) + 24*jnp.exp(t*33*1j)*jnp.exp(3*x) + 54*jnp.exp(t*33*1j)*jnp.exp(5*x) + 3*jnp.exp(t*25*1j)*jnp.exp(17*x) + 54*jnp.exp(t*33*1j)*jnp.exp(13*x) + 24*jnp.exp(t*33*1j)*jnp.exp(15*x) + 135*jnp.exp(t*41*1j)*jnp.exp(9*x) + 30*jnp.exp(t*49*1j)*jnp.exp(5*x) + 120*jnp.exp(t*49*1j)*jnp.exp(7*x) + 120*jnp.exp(t*49*1j)*jnp.exp(11*x) + 30*jnp.exp(t*49*1j)*jnp.exp(13*x) + 60*jnp.exp(t*57*1j)*jnp.exp(9*x)))/(3*(jnp.exp(t*24*1j) + 10*jnp.exp(6*x) + 10*jnp.exp(12*x) + 45*jnp.exp(t*8*1j)*jnp.exp(8*x) + 45*jnp.exp(t*8*1j)*jnp.exp(10*x) + 18*jnp.exp(t*16*1j)*jnp.exp(4*x) + 9*jnp.exp(t*24*1j)*jnp.exp(2*x) + 18*jnp.exp(t*16*1j)*jnp.exp(14*x) + 64*jnp.exp(t*24*1j)*jnp.exp(6*x) + 36*jnp.exp(t*24*1j)*jnp.exp(8*x) + 36*jnp.exp(t*24*1j)*jnp.exp(10*x) + 64*jnp.exp(t*24*1j)*jnp.exp(12*x) + 18*jnp.exp(t*32*1j)*jnp.exp(4*x) + 9*jnp.exp(t*24*1j)*jnp.exp(16*x) + jnp.exp(t*24*1j)*jnp.exp(18*x) + 18*jnp.exp(t*32*1j)*jnp.exp(14*x) + 45*jnp.exp(t*40*1j)*jnp.exp(8*x) + 45*jnp.exp(t*40*1j)*jnp.exp(10*x) + 10*jnp.exp(t*48*1j)*jnp.exp(6*x) + 10*jnp.exp(t*48*1j)*jnp.exp(12*x)))  
        
        
        return ut

    def exact_soln_real(t,x,q):
        return jnp.real(NLS_True_Sol(t,x,q))    
    def exact_soln_imag(t,x,q):
        return jnp.imag(NLS_True_Sol(t,x,q))    
    def exact_soln_np(t,x,q):
        return np.array(NLS_True_Sol(t,x,q))



    t_ini = 0.0
    xj = jnp.array(x)
    sol_real_x =  grad(exact_soln_real,1)
    sol_imag_x =  grad(exact_soln_imag,1)

    sol_real_xx =  grad(sol_real_x,1)
    sol_imag_xx =  grad(sol_imag_x,1)

    #gsol_real =  grad(exact_soln_real,1)
    #gsol_imag =  grad(exact_soln_imag,1)
    sol_real_x_vm =  jax.vmap(grad(exact_soln_real,1),(0,0,None))
    sol_imag_x_vm =  jax.vmap(grad(exact_soln_imag,1),(0,0,None))

    sol_real_xx_vm =  jax.vmap(grad(sol_real_x,1),(0,0,None))
    sol_imag_xx_vm =  jax.vmap(grad(sol_imag_x,1),(0,0,None))

    dx_sol_real=sol_real_x_vm(t_ini*np.ones_like(x),xj,q)
    dx_sol_imag=sol_imag_x_vm(t_ini*np.ones_like(x),xj,q)
    dx_sol = np.array(dx_sol_real)+1j*np.array(dx_sol_imag)

    dxx_sol_real=sol_real_xx_vm(t_ini*np.ones_like(x),xj,q)
    dxx_sol_imag=sol_imag_xx_vm(t_ini*np.ones_like(x),xj,q)
    dxx_sol = np.array(dxx_sol_real)+1j*np.array(dxx_sol_imag)

    sol_real = exact_soln_real(t_ini*np.ones_like(x),xj,q)
    sol_imag = exact_soln_imag(t_ini*np.ones_like(x),xj,q)
    sol = np.array(sol_real)+1j*np.array(sol_imag)

    amp = np.square(sol_real)+np.square(sol_imag)
    npsol = np.square(np.abs(exact_soln_np(np.zeros_like(x),x,q)))

    print(type(sol),sol.dtype)



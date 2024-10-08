import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

import numpy as np



if __name__=="__main__":
    kppa = 8.0
    a= kppa*kppa/16.0
    c = 0.5
    xo = -2.5
    



    L = 2*20.0; m = 512
    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    T = 5.0


    def exact_soln_np(t,x,kppa,xo=xo):
        bta=-kppa
        omg = np.sqrt(a)*((x-xo)-c*t)
        #print("omg shape",omg.shape)
        f = (2.0*a/bta)/np.cosh(omg)
        thta = 0.5*c*(x-xo) - (0.25*c*c -a)*t
        
        sol = f*np.cos(thta)+1j*f*np.sin(thta)
        
        return(sol)

    def exact_soln_real(t,x,a=a,c=c,xo=xo,kppa=kppa):
        bta = - kppa
        omg = np.sqrt(a)*((x-xo)-c*t)
        #print("omg shape",omg.shape)
        f = (2.0*a/bta)/jnp.cosh(omg)
        thta = 0.5*c*(x-xo) - (0.25*c*c -a)*t
        
        sol = f*jnp.cos(thta)
        
        return(sol)
    def exact_soln_imag(t,x,a=a,c=c,xo=xo,kppa=kppa):
        bta = - kppa
        omg = np.sqrt(a)*((x-xo)-c*t)
        #print("omg shape",omg.shape)
        f = (2.0*a/bta)/jnp.cosh(omg)
        thta = 0.5*c*(x-xo) - (0.25*c*c -a)*t
        
        sol = f*jnp.sin(thta)
        
        return(sol)

    #sol = jax.vmap(exact_soln)

    t_ini = 0.0
    xj = jnp.array(x)
    #gsol_real =  grad(exact_soln_real,1)
    #gsol_imag =  grad(exact_soln_imag,1)
    gsol_real =  jax.vmap(grad(exact_soln_real,1),(0,0))
    gsol_imag =  jax.vmap(grad(exact_soln_imag,1),(0,0))

    dx_sol_real=gsol_real(t_ini*np.ones_like(x),xj)
    dx_sol_imag=gsol_imag(t_ini*np.ones_like(x),xj)
    dx_sol = np.array(dx_sol_real)+1j*np.array(dx_sol_imag)

    sol_real = exact_soln_real(t_ini*np.ones_like(x),xj)
    sol_imag = exact_soln_imag(t_ini*np.ones_like(x),xj)
    sol = np.array(sol_real)+1j*np.array(sol_imag)
    print(type(sol),sol.dtype)


    amp = np.square(sol_real)+np.square(sol_imag)
    npsol = np.square(np.abs(exact_soln_np(np.zeros_like(x),x,kppa)))

    #plt.plot(xj,-np.real(sol),"r-")
    #plt.plot(xj,np.real(dx_sol),"r--")
    #plt.plot(xj,np.imag(sol),"b-")
    #plt.plot(xj,np.imag(dx_sol),"b--
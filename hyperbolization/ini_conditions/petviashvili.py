
import pickle
import numpy as np
import sys
if __name__=="__main__":
    #m = 1024
    T=5.0
    tau_name = sys.argv[1]
   
    file_name="Petviashvilil_"+tau_name+"_ini.pkl"
    
    with open(file_name, 'rb') as f:
        fdict = pickle.load(f)
    v = fdict["v"]
    p = fdict["p"]
    #x = fdict["x"]
    #xi = fdict["xi"]
    kppa = fdict["kppa"]
    mu = fdict["mu"]
    L = fdict["L"]

    #m = v.shape[0]
    skip = 32  # or 32
    m = int(2**15/skip)
    print("m is ",m)
    

    sol  = np.exp(1j*mu*0.0)*v
    dx_sol = np.exp(1j*mu*0.0)*p
    #skip = int(sol.shape[0]/m)

    sol = sol[::skip]
    dx_sol = dx_sol[::skip]
    #x = x[::skip]
    #xi= xi[::skip]

    v = v[::skip]
    p = p[::skip]



    def exact_soln_np(t,x,kppa,v=v,p=p):
        exp_fac = np.exp(1j*mu*t)
        q0 = exp_fac*v
        q1 = exp_fac*p
        return q0
    
    print("Ini Pits",file_name,"tau name",tau_name)
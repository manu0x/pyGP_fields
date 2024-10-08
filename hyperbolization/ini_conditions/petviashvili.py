
import pickle
import numpy as np
if __name__=="__main__":
    m = 512

    file_name="Petviashvili_ini.pkl"
    with open(file_name, 'rb') as f:
        fdict = pickle.load(f)
    v = fdict["v"]
    p = fdict["p"]
    x = fdict["x"]
    xi = fdict["xi"]
    kppa = fdict["kppa"]
    mu = fdict["mu"]

    L = x[-1]-x[0]
    

    

    sol  = np.exp(1j*mu*0.0)*v
    dx_sol = np.exp(1j*mu*0.0)*p
    skip = int(sol.shape[0]/m)

    sol = sol[::skip]
    dx_sol = dx_sol[::skip]
    x = x[::skip]
    xi= x[::skip]

    v = v[::skip]
    p = p[::skip]



    def exact_soln_np(t,x,kppa,v=v,p=p):
        exp_fac = np.exp(1j*t)
        q0 = exp_fac*v
        q1 = exp_fac*p
        return q0
import os
import sys
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve   
from scipy.optimize import elementwise

cwd = "/".join(os.getcwd().split("/")[:-2])
sys.path.append(cwd)
print("CWD:",cwd)


from GPE import GPE_scalar_field_1d2c
from GPE import ImEx

from SP_2d_functions_B import *



def laplacian(u,xi2):
     ft_u = -np.fft.fftn(u,u.shape)*xi2
     #ft_u[0,0] = 0.0
     return np.fft.ifftn(ft_u,u.shape)



if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

   

   
    imex_sch = str(sys.argv[1])
    relax = int(sys.argv[2])  ## 0- No relax, 1- mass only relax, 2- full relax
    energy_type= sys.argv[3]  ## 
    A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages = choose_imex(imex_sch)  
    

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex,emb_B=b_hat,im_C=C,ex_C=C)


    # List of tau values for which u wanna run simulation
    
    
    frame_dict_list = []
  

    t_list = []

    # Effective value for reduced Planck's constant divided by mass (hbar/m)
    hbar_mass = 6e-5
    beta = 1.5
    epsilon = hbar_mass

    

    kppa = beta/(epsilon)  ## Coefficient in front of non-linear term

    lap_fac = epsilon/(2.0)  ## Coefficient in front of laplacian term
    
    dt=5e-5     ## Choose dt
    t_ini = 0.01
    T = 0.088
    print("T",T,"energy type",energy_type,"dt",dt,"type dt",type(dt))

    m = 1024  ## Number of grid points
    xL = -0.5; xR = 0.5; L = xR-xL
    Lf = 1.0
    x_1d = np.arange(-m/2,m/2)*(L/m)
    x,y = np.meshgrid(x_1d,x_1d)

    xi = np.fft.fftfreq(m)*m*2*np.pi/L
    xix,xiy = np.meshgrid(xi,xi)

    xi2 = xix*xix + xiy*xiy


    
    case="imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    save_dir = "_data_B_inifile_"+energy_type#"./_data_B_inifile_En_"+str(energy_type)
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    save_dir = save_dir+"/"+case
   
    
    
    
    def D_func(t):
          return t
    

    def ini_f_solve(q,x,A1,A2,D,L,prtn_1,prtn_2):
          #print("xsape",q.shape,x.shape)
        

    
          res = np.square(q*prtn_1 - D*A1*(L/np.pi)*np.sin(q*prtn_1*np.pi/L) - x*prtn_1) + np.square(q*prtn_2 - D*A2*(L/np.pi)*np.sin(q*prtn_2*np.pi/L) - x*prtn_1)
          #print("res shape",res.shape,q.max(),q.min())
          return res
    # def ini_f_jacob(q,x,A1,A2,D,L,prtn_1,prtn_2):

    #     cord1 = 2.0*(q*prtn_1 - D*A1*(L/np.pi)*np.sin(q*prtn_1*np.pi/L)- x*prtn_1) *(prtn_1- D*A1*prtn_1*np.cos(q*prtn_1*np.pi/L)) 
    #     cord2 = 2.0*(q*prtn_2 - D*A2*(L/np.pi)*np.sin(q*prtn_2*np.pi/L)- x*prtn_2) *(prtn_2- D*A2*prtn_2*np.cos(q*prtn_2*np.pi/L)) 
    #     return np.diag(cord1+cord2)

    # def Q(x,A,D,L,flat_size):
    #         print("Q check",type(x),x.shape)
    #         q_0 =np.concatenate([x[:,:,0].flatten(),x[:,:,1].flatten()])
    #         x1 = x[:,:,0].flatten()
    #         x2 = x[:,:,1].flatten()
    #         x = np.concatenate([x1,x2])
    #         A1 = A[0]
    #         A2 = A[1]
    #         prtn_1 =  np.concatenate([1.0*x1,0.0*x1])
    #         prtn_2 =  np.concatenate([0.0*x1,1.0*x1])
    #         print("IIxsape",q_0.shape,A1,A2)
    #         q,info,ier,msg = fsolve(ini_f_solve,q_0,fprime = ini_f_jacob,args=(x,A1,A2,D,L,prtn_1,prtn_2),full_output=True,xtol=1e-14)
    #         print("q shape 1",q.shape,q.min(),q.max(),ier,msg)
    #         q_1 = q[:flat_size]
    #         q_2 = q[flat_size:]
    #         q_1 = np.reshape(q_1,[m,m])
    #         q_2 = np.reshape(q_2,[m,m])

    #         return np.stack([q_1,q_2],axis=-1)
    
    def f_element(q,x,w,l):
         #print(q.max())
         return q-w*np.sin(q*l)-x
    
    def Q_element(x,A,D,L,flat_size):
        
        q = []
        for i in range(2):
            print("Doing component ",i)
            xc = x[:,:,i].flatten()
            Ac = A[i]
            w = D*Ac*(L/np.pi)
            l = np.pi/L
            res_bracket = elementwise.bracket_root(f_element, xl0 = -2.0*np.ones_like(xc), xr0 = 2.0*np.ones_like(xc), args=(xc,w,l),maxiter=2000)
            print("Resc",res_bracket.bracket)
            #print("Res Brckatre",res_bracket.success)
            res_root = elementwise.find_root(f_element, res_bracket.bracket,args=(xc,w,l))
            print("f value",res_root.f_x.max(),res_root.x.max(),res_root.x.min())
           
            q.append(np.reshape(res_root.x,[m,m]))
          
        return np.stack(q,axis=-1)
    
    
          

    def initial_conditions(x,t_ini,D_func,L,xi2):
        ###############################################################################################
        # A1 = 30.
        # A2 = 40.
        # A =np.stack([A1,A2],axis=-1)
        # D = D_func(t_ini)
        # #q = Q(x,A,D,L,flat_size=m*m)
        # q = Q_element(x,A,D,L,flat_size=m*m)
        
        # n_d = 1.0/((1.0-D*A1*np.cos(q[:,:,0]*np.pi/L))*(1.0-D*A2*np.cos(q[:,:,1]*np.pi/L)))
        # #n_d = n_d/np.mean(n_d)
        # H = np.power(t_ini,-1.5)
        # f = 1.0
        # phi_d = t_ini*t_ini*H*f*D*(A1*(L/np.pi)*(L/np.pi)*np.cos(q[:,:,0]*np.pi/L)+A2*(L/np.pi)*(L/np.pi)*np.cos(q[:,:,1]*np.pi/L)   \
        #                            + 0.5*D*( (A1*(L/np.pi)*np.sin(q[:,:,0]*np.pi/L))**2.0 + (A2*(L/np.pi)*np.sin(q[:,:,1]*np.pi/L))**2.0 ) )  
        

        ################################.   Loading from initial data file     ###########################################

        data = np.loadtxt("./x_y_n_phi_1024.dat")
        data = np.reshape(data,(1024,1024,2))
        n_d = data[::,::,0]
        phi_d = data[::,::,1]
        print("n_d from file",n_d.max(),n_d.min(),phi_d.max(),phi_d.min())
        ###########################################################################
        #N = len(x)
        #### Argument to exponential 

        u_exp_arg = 1j*phi_d/hbar_mass
        u_ini = np.sqrt(n_d)*np.exp(u_exp_arg)
        print("Type check",u_exp_arg.dtype,n_d.max(),n_d.min())
        
        #u_ini = u_ini/np.mean(np.abs(u_ini)**2)
        print("u_ini shape:",n_d.min(),n_d.max(),u_ini.shape,L*L*np.mean(np.square(np.abs(u_ini))),L)
        # from matplotlib import cm
        # ##################################################################
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(3, 2, 1, projection='3d')
        # #ax.set_box_aspect(1)
        # #ax.pcolormesh(x[:,:,0],x[:,:,1],np.square(np.abs(u_ini)))
        # ax.plot_surface(x[:,:,0],x[:,:,1],np.square(np.abs(u_ini)), cmap=cm.plasma,
        #                linewidth=0, antialiased=False)
        # #plt.plot(x[:,:,0].flatten(),q[:,:,0].flatten())
        # #plt.plot(x[:,:,1].flatten(),q[:,:,1].flatten())
        # #splt.colorbar()


        # ax = fig.add_subplot(3, 2, 2)
        # ax.set_box_aspect(1)
        # pclm = ax.pcolormesh(x[:,:,0],x[:,:,1],n_d)
     
        # fig.colorbar(pclm,ax=ax)
        # ##################################################################
        # ax = fig.add_subplot(3, 2, 3, projection='3d')
        # ax.plot_surface(x[:,:,0],x[:,:,1],phi_d, cmap=cm.plasma,
        #                linewidth=0, antialiased=False)
        # #plt.plot(x[:,:,0].flatten(),q[:,:,0].flatten())
        # #plt.plot(x[:,:,1].flatten(),q[:,:,1].flatten())
        # #splt.colorbar()

        # ax = fig.add_subplot(3, 2, 4)
        # ax.set_box_aspect(1)
        # pclm = ax.pcolormesh(x[:,:,0],x[:,:,1],phi_d)
     
        # fig.colorbar(pclm,ax=ax)
        # ##################################################################
        # ax = fig.add_subplot(3, 2, 5, projection='3d')
       
      
        # div_u = laplacian(phi_d+1j*np.zeros_like(phi_d),xi2).real

        # ax.plot_surface(x[:,:,0],x[:,:,1],div_u, cmap=cm.plasma,
        #                linewidth=0, antialiased=False)
        # #plt.plot(x[:,:,0].flatten(),q[:,:,0].flatten())
        # #plt.plot(x[:,:,1].flatten(),q[:,:,1].flatten())
        # #splt.colorbar()

        # ax = fig.add_subplot(3, 2, 6)
        # ax.set_box_aspect(1)
        # pclm = ax.pcolormesh(x[:,:,0],x[:,:,1],div_u)
     
        # fig.colorbar(pclm,ax=ax)

        # plt.savefig("initial_cond_B_from_file.png")
        return u_ini

    q0_ini  = initial_conditions(np.stack([x,y],axis=-1),t_ini,D_func,Lf,xi2=xi2)
    
    #file_n = "./_data/imex_ARK3(2)4L[2]SA_" + str(m) + "_5e-05_zero_vel_relaxed/frame_70.npz"
    #q0_ini = np.load(file_n)['frame']
    q0_ini_mass = np.mean(np.square(np.abs(q0_ini)))*L*L
    print("Initial mass is",q0_ini_mass,L)
    #q0_ini = q0_ini/np.sqrt(q0_ini_mass)
    print("q0_ini",q0_ini.shape,q0_ini.dtype)
    # q1_ini  = np.zeros_like(q0_ini)
    u_ini = q0_ini#np.stack((q0_ini,q1_ini),axis=1)

  

    print("Running Relaxed SP_2d Example 1 with scheme ",imex_sch)
   
    if relax>0:
        if relax==1:
            save_dir_case = save_dir+"_relaxed_massOnly"
        else:
            save_dir_case = save_dir+"_relaxed"
    else:
        save_dir_case = save_dir
    if not(os.path.exists(save_dir_case)):
            os.makedirs(save_dir_case)
    frm,tt,inv_change_dict,mass_l,mass_err_l,energy_err_l,energy_Fd_l = run_SP_2d_example_1(energy_type,dt,[x,y],[xix,xiy],kppa,t_ini,T,L,imx,[],u_ini,exact_soln_np=None,dx_soln_jx=None,\
                                                            relax=relax,log_errs=False,lap_fac=lap_fac,num_plots=100,p=None,data_dir=save_dir_case)
    case_dict={"scheme":imex_sch,"frame_list":frm,"t_list":tt,"inv_change_dict":inv_change_dict,"x":[x,y],"xi":[xix,xiy],\
                   "kappa":kppa,"dt":dt,"m":m,"mass_l":mass_l,"mass_err_l":mass_err_l,"energy_err_l":energy_err_l,"energy_FD_l":energy_Fd_l}
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
            pickle.dump(case_dict,f)
    


    
    
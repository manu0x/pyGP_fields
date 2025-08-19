import numpy as np

class GPE_scalar_field_1d2c_relax:
    def __init__(self,N,nc=2,im_rhs=None,ex_rhs=None,imx=None,ini_psi=None,relax=False,tau=1.0):
        self.my_shape=()
        self.s = imx.s
        self.dim = 1
        self.tau = tau

        '''Relaxation related'''
        self.relax=relax
        self.rel_gamma = 1.0
        self.rel_num_sum = 0.0
        self.rel_den_sum = 0.0
        self.term1 = 0.0
        
        self.my_shape=(N,nc)
        self.K_shape = (self.s,)+self.my_shape

        #print("class shapes",self.my_shape,self.K_shape)
        #print("Using Relaxation")
        
        
        self.psi = np.zeros(self.my_shape,dtype=np.complex128)
        self.psi = 1.0*ini_psi

        self.mass_ini = np.sum(np.abs(ini_psi)**2) 
        
        #print("my shape",self.my_shape,"psi shape",self.psi.shape)
        
        self.f = np.zeros_like(self.psi)
        self.f_t = np.zeros_like(self.f)
        
        self.f = 1.0*self.psi
        
        self.im_K = np.zeros(self.K_shape,dtype=np.complex128)
        self.ex_K = np.zeros_like(self.im_K)
        
        
        self.ex_rhs = ex_rhs
        self.im_rhs = im_rhs
        
        self.im_A = imx.im_A
        self.ex_A = imx.ex_A
        
        self.im_B = imx.im_B
        self.ex_B = imx.ex_B
        
        self.im_C = imx.im_C
        self.ex_C = imx.ex_C
    
    def do_fft(self,s_cntr,lmda,dt):
        self.f_t[:,0] = np.fft.fft(self.f[:,0])
        self.f_t[:,1] = np.fft.fft(self.f[:,1])
    
        #(1+i*dt*a[s][s]*lmda)*f_t = ft(rhs)
        self.f_t = np.squeeze(np.matmul(lmda,np.expand_dims(self.f_t,axis=-1)))
        self.f[:,0] = np.fft.ifft(self.f_t[:,0])
        self.f[:,1] = np.fft.ifft(self.f_t[:,1])
        #print("f shape",self.f_t.shape,self.f.shape)
        
    def update_stage_sum(self,s_cntr,dt):
        for i in range(s_cntr):
            if(i==0):
                self.f = self.psi + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            else:
                self.f = self.f + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            
    def update_K(self,s_cntr,*args):
        #print("sncntr ",s_cntr)
       # print("psi shape",self.psi.shape,"f shape",self.f.shape)
        if (s_cntr==0)and(self.relax):
            self.rel_num_sum=0.0
        self.ex_K[s_cntr,:] = self.ex_rhs(self.f,self.f_t,*args)
        self.im_K[s_cntr,:] = self.im_rhs(self.f_t,self.f,*args)
        #if(self.relax):
            #self.rel_num_sum+= np.sum(np.abs(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*(self.f-self.psi)))
         #   self.rel_num_sum+= (np.sum(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr,:,0]+ self.im_B[s_cntr]*self.im_K[s_cntr,:,0])*(self.f[:,0]-self.psi[:,0]))\
        #                        +self.tau*np.sum(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr,:,0]+ self.im_B[s_cntr]*self.im_K[s_cntr,:,0])*(self.f[:,1]-self.psi[:,1])) )
        
    def sum_contributions(self,dt):
        term = np.zeros_like(self.psi)
        term_rel_den = 0.0
        for i in range(self.s):
            term+= (dt*self.ex_B[i]*self.ex_K[i]+ dt*self.im_B[i]*self.im_K[i])
            term_rel_den+= (self.ex_B[i]*self.ex_K[i]+ self.im_B[i]*self.im_K[i])
            
        if (self.relax):
            self.rel_den_sum = np.sum(np.conj(term_rel_den[:,0])*term_rel_den[:,0])+self.tau*(np.sum(np.conj(term_rel_den[:,1])*term_rel_den[:,1]))

            self.term1 = (np.sum( np.conj(term_rel_den[:,0])*self.psi[:,0]+np.conj(self.psi[:,0])*term_rel_den[:,0] +\
                                  self.tau*( np.conj(term_rel_den[:,1])*self.psi[:,1]+np.conj(self.psi[:,1])*term_rel_den[:,1] )  ))
            
            self.rel_gamma = np.real(1.0*(-self.term1/(dt*self.rel_den_sum)))
            #print("ter",term_rel_den.shape,self.rel_den_sum.shape)
            #print("rrRel gamma is ",self.rel_gamma,2.0*dt*self.rel_num_sum/self.rel_den_sum,self.rel_den_sum)

        
        mass_old = np.sum(np.conj(self.psi[0])*self.psi[0])+ self.tau*np.sum(np.conj(self.psi[1])*self.psi[1])
        self.psi = self.psi + self.rel_gamma*term
        mass_new = np.sum(np.conj(self.psi[0])*self.psi[0])+ self.tau*np.sum(np.conj(self.psi[1])*self.psi[1])

        #print("mass change",(mass_new-mass_old).real,self.tau)
            
        self.f = 0.0+self.psi
        
    def calc_mass(self):
        return(np.sum(np.abs(self.psi)**2).flatten())
    

        
        #im_rhs takes only fourier space vector 
        
       







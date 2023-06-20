import numpy as np

class GPE_scalar_field:
    def __init__(self,dim,N,im_rhs=None,ex_rhs=None,imx=None,ini_psi=None):
        self.my_shape=()
        self.s = imx.s
        for i in range(dim):
            self.my_shape=self.my_shape+(N,)
        self.K_shape = (self.s,)+self.my_shape
        
        
        
        self.psi = np.zeros(self.my_shape,dtype=np.complex64)
        self.psi = 1.0*ini_psi
        
        print(self.my_shape,self.psi.shape)
        
        self.f = np.zeros_like(self.psi)
        self.f_t = np.zeros_like(self.f)
        
        self.f = 1.0*self.psi
        
        self.im_K = np.zeros(self.K_shape,dtype=np.complex64)
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
        self.f_t = np.fft.fftn(self.f,self.my_shape)
    
        #(1+i*a[s][s]*lmda)*f_t = ft(rhs)
        self.f_t = self.f_t/(1.0+1j*dt*lmda*self.im_A[s_cntr][s_cntr])
        self.f = np.fft.ifftn(self.f_t,self.my_shape)
        
    def update_stage_sum(self,s_cntr,dt):
        for i in range(s_cntr):
            if(i==0):
                self.f = self.psi + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            else:
                self.f = self.f + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            
    def update_K(self,s_cntr,*args):
        #print("sncntr ",s_cntr)
       # print("psi shape",self.psi.shape,"f shape",self.f.shape)
        self.ex_K[s_cntr,:] = self.ex_rhs(self.f,*args)
        self.im_K[s_cntr,:] = self.im_rhs(self.f_t,args[0])
        
    def sum_contributions(self,dt):
        for i in range(self.s):
            self.psi = self.psi + dt*self.ex_B[i]*self.ex_K[i]+ dt*self.im_B[i]*self.im_K[i]
            
        self.f = 0.0+self.psi
        
        
        #im_rhs takes only fourier space vector 
        
       







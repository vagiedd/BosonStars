import numpy as np
import matplotlib.pyplot as plt
#import numdifftools as nd
import scipy as scipy
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
import warnings

class relaxation_method:
    def __init__(self,xmin,xmax,M,N,B1=[],B2=[],y_guess=[],iteration=100,start=False,display=False,**kwargs):
        #boundary conditions N=n1+n2
        self.N = N
        self.M = M
        self.n1 = len(B1)
        self.n2 = len(B2)
        self.B1 = B1
        self.B2 = B2
        
        self.iteration = iteration
        self.display = display
        self.k = 0
        self.slowc = 1.0
        self.successful = True
        self.eps = 1.0e-10
        self.tol = 1.0e-15

        #mesh grid
        self.x = np.linspace(xmin,xmax,M)
        # initial guess of y[i,j], stored in 2D array
        self.y = np.ones((N,M))     
        self.set_initial_guess = False
        if(len(y_guess)!=0):
            if(y_guess.shape!=self.y.shape):
                exit("shape of y does not match N x M shape")
            else: 
                self.set_initial_guess = True
                self.y = y_guess
        #error
        self.err = []
        self.iterations = []

        self.init(**kwargs)
        if(start): self.x,self.y = self.start()

        self.dic = dict(err=self.err,iterations=self.iterations,successful=self.successful)


    def init(self,*args,**kwargs):
        pass

    def f(self,t,y,i=None):
        """
        #x = t
        #y[0] = x'
        #y[1] = x
        #y[2] = y
        #f = [2-x, y*x, z]
        x = y.item(0)
        u = y.item(1)
        z = y.item(2)
        f = [z,2.0-x,u*x]
        if(i is None): return np.array(f)
        else: return f[i]
        """
        return 0

    def jac(self,t,y,i=None,j=None):
        """
        #x = t
        #y[0] = x
        #y[1] = y
        #y[2] = z = x'
        x = y.item(0)
        u = y.item(1)
        z = y.item(2) 
        jac = np.zeros((self.N,self.N),dtype=np.float64)

        jac[0][0] = 0
        jac[0][1] = 0
        jac[0][2] = 1

        jac[1][0] = -1.0
        jac[1][1] = 0
        jac[1][2] = 0

        jac[2][0] = u
        jac[2][1] = x
        jac[2][2] = 0
        
        if(i is None):
            return jac
        else: return jac.item(i,j)
        """
        #return nd.Jacobian(lambda y,x: self.f(x,y),order=2)((y[...,k]+y[...,k-1])/2.,(x[k]+x[k-1])/2.)
        eps = self.eps
        N = len(y)
        jac = np.zeros([N,N], dtype = np.float64)     
        for i in range(N):         
            y1 = y.copy()        
            y2 = y.copy()          
            y1[i] += eps         
            y2[i] -= eps          
            f1 = self.f(t, y1)         
            f2 = self.f(t, y2)          
            jac[ : , i] = (f1 - f2) / (2 * eps)      
        return jac

    def dB1dy(self):
        #shape n1 x N
        dB1dy = np.zeros((self.n1,self.N))
        """
        dB1dy[0][0] = 1.0
        dB1dy[1][1] = 1.0
        """
        return dB1dy

    def dB2dy(self):
        #shape n2 x N
        dB2dy = np.zeros((self.n2,self.N))
        """
        dB2dy[0][0] = 1.0
        """
        return dB2dy

    def scale(self,k):  
        return 0.25

    def delta(self,i,j):
        if(i==j):
            return 1.0
        else: 
            return 0.0

    def CalcError(self,deltaylist): 
        errSum=0
        for ip in range(self.M):
            for ie in range(self.N): 
                errSum = errSum + abs(deltaylist[ip*self.N+ie])/self.scale(ie)
        return errSum/(self.M*self.N)

    def CalcRes(self):
        n1 = self.n1
        n2 = self.n2
        N = self.N
        M = self.M
        x = self.x
        y = self.y
        f = self.f
        B1 = self.B1
        B2 = self.B2
        
        reslist=[]
        # first ne residuals at k=0
        for i in range(n1): 
            E = y[i][0] - self.B1[i]
            if(self.set_initial_guess):
                reslist.append(0)
            else: 
                reslist.append(E)
        # for each k=1,..,M-1, there will be 2ne residuals
        for k in range(1,M):
            E = y[...,k] - y[...,k-1] - (x[k]-x[k-1])*(self.f(.5*(x[k]+x[k-1]),.5*(y[...,k]+y[...,k-1])))
            for j in range(N):
                reslist.append(E.item(j))
        # last ne residuals at k=M
        for i in range(n2): 
            E = y[i][-1] - self.B2[i]
            if(self.set_initial_guess):
                reslist.append(0)
            else: reslist.append(E)
       
        return np.array(reslist,dtype=np.float64)

    def UpdateY(self,deltayList):
        for k in range(self.M): 
            for j in range(self.N): self.y[j,k]=self.y[j,k]+deltayList[k*self.N+j]

    def CalcS(self):
        n1 = self.n1
        n2 = self.n2
        N = self.N
        M = self.M
        x = self.x
        y = self.y
        s = np.zeros((M*N, M*N),dtype=np.float64)
        ns = M*N
        #at initial boundary
        for i in range(n1):
            for j in range(N):
                s[i][j] = self.dB1dy()[i][j]
        #at outer boundary 
        for i in range(n2):
            for j in range(N):
                s[ns-n2+i][(M-1)*N+j] = self.dB2dy()[i][j]
        #interior points 
        for k in range(1,M):
            dx = x[k] - x[k-1]
            r0 = n1 + N*(k-1)
            c0 = N*(k-1)
            self.k = k
            dgdy = self.jac((x[k]+x[k-1])/2.,(y[...,k]+y[...,k-1])/2.)
            for i in range(N):
                for j in range(N):
                    s[r0+i][c0+j] = -self.delta(i,j) - 0.5*dx*dgdy.item(i,j)
            r1 = r0
            c1 = c0 + N
            for i in range(N):
                for j in range(N):
                    s[r1+i][c1+j] = self.delta(i,j) - 0.5*dx*dgdy.item(i,j)
        return np.array(s)

    def start(self):
        errList = []
        itList = []
        for it in range(self.iteration):
            resList=self.CalcRes()
            s = self.CalcS()
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore")
                deltayList = spsolve(s, -resList)
            err = self.CalcError(deltayList)
            if(self.display): print(it,err)
            errList.append(err)
            itList.append(it)
            stopIteration = err < self.tol#pow(10,-15)
            if not stopIteration: deltayList = self.slowc/max([self.slowc,err])*deltayList
            self.UpdateY(deltayList)
            if stopIteration: break
            if it == self.iteration-1:
                self.successful = False
                break 
        self.err = errList
        self.iterations = itList
        return self.x, self.y


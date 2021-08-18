import numpy as np
import scipy as scipy
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
import warnings

"""
    The primary task of this module is to solve a set of differential equations subject to boundary conditions.
    Implements the relaxation algorithm given in "Numerical Recipies in C: The Art of Scientific Computing". 
    The method converts the ODEs to Finite-Difference equations (FDEs). Please see 
    "Numerical Recipies in C: The Art of Scientific Computing" for further details. 
"""

class relaxation_method():
    """
        An abstract class that implements the relaxation method. The following functions should be overriden

        :func: init function to change attributes

        :func: f defines the differential equations of the equation of motion. should be overridden

        :func: jac defines the jacobian of the differential equations. can be overridden otherwise
        use finite differences to compute the jacobian.  

        Parameters:
        -----------
        xmin : float
            The minimum value of the grid defined at the left boundary
        xmax : float
            The maximum value of the grid defined at the right boundary
        M : int
            Length of the grid
        N : int
            Number of differential equations
        B1 : list
            Left boundary conditions at xmin
        B2 : list
            Right boundary conditions at xmax
        y_guess : list
            Initial guess for the relaxation algorithm. Filled with ones if empty
        iteration : int
            Maximum number of iterations to run the relaxation algorithm
        start : bool
            Whether to start the relaxation immediately once an object of relaxation_method is created
        display : bool
            Print out iterations of the relaxation
        kwargs : dictionary
            Optional parameters to pass to init if it is overridden 
    """
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

    def f(self,t,y):
        """
        Differential equations of the form 
            dy/dx = g(x, y)
        See tests directory for examples or bosonstar.py

        Parameters:
        -----------
            t : float 
                position on mesh grid
            y : numpy array 
                derivatives of the variables in the differential equations

        """
        pass

    def jac(self,t,y,eps=1e-8):
        """
        Jacobian of the differential equations of the EKG system.
        Used in the relaxation algorithm.
        See tests directory for examples or bosonstar.py

        Parameters:
        -----------
            t : float 
                position on mesh grid
            y : numpy array 
                derivatives of the variables in the differential equations

        """
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
        """
        Boundary Conditions at xmin
        Shape n1 x N

        Attributes: These are initialized during the construction of the class object
        -----------
            n1 : int 
                number of boundary conditions at r = 0
            N : int
                number of differential equations
        """
        #shape n1 x N
        dB1dy = np.zeros((self.n1,self.N))
        return dB1dy

    def dB2dy(self):
        """
        Boundary Conditions at xmin
        Shape n2 x N

        Attributes: These are initialized during the construction of the class object
        -----------
            n2 : int 
                number of boundary conditions at r = 0
            N : int
                number of differential equations
        """
        #shape n2 x N
        dB2dy = np.zeros((self.n2,self.N))
        return dB2dy

    def delta(self,i,j):
        """
        Kronecker delta function 

        Parameters:
        -----------
            i, j : int
                positions in a matrix

        Returns:
        --------
        1 or 0
        """
        if(i==j):
            return 1.0
        else: 
            return 0.0

    def scale(self,k):  
        """
        Scale of the corrections used in the relaxation algorithm associated with each of the differential equations

        Parameters:
        -----------
            k : int
                Specifies the position of the scale array

        Returns:
        --------
            scale : float or list
                parameter k gives the position in scale if it is a list
            
        """
        scale = 0.25
        return scale

    def CalcError(self,deltaylist): 
        """
        Computes the value of the average correction by summing the absolute value of all corrections,
        weighted by a scale factor appropriate to each type of variable. This is used to check for 
        convergance 

        Parameters:
        ----------
            deltaylist : numpy array
                Corrections to the variables in the differential equations

        Returns:
        --------
            err : float
                Error to the corrections
        """
        errSum=0
        for ip in range(self.M):
            for ie in range(self.N): 
                errSum = errSum + abs(deltaylist[ip*self.N+ie])/self.scale(ie)
        return errSum/(self.M*self.N)

    def CalcRes(self,x,y):
        """"
        This function calculates the FDEs of the differential equations

        Parameters:
        -----------
            x : numpy array
                Mesh grid
            y : numpy array
                solutions for the variables in the differential equations

        Returns:
        --------
            reslist : numpy array 
                The FDEs at the boundary conditions and the interior of the mesh grid
        """
        n1 = self.n1
        n2 = self.n2
        N = self.N
        M = self.M
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

    def UpdateY(self,y,deltayList):
        """
        Update the results of the variables after solving the finite difference equations and computing the corrections

        Parameters:
        -----------
            y : numpy array
                solutions for the variables in the differential equations

        Returns:
        --------
            y : numpy array
                updates to y

        """
        for k in range(self.M): 
            for j in range(self.N): y[j,k]=y[j,k]+deltayList[k*self.N+j]
        return y

    def CalcS(self,x,y):
        """
        Computes the derivatives of the residules from :func: CalcE with respect to y

        Parameters:
        -----------
            x : numpy array
                mesh grid
            y : numpy array
                solutions for the variables in the differential equations

        Returns:
        --------
            S : numpy array
                matrix of the form dE/dy where E is the residuals from finite difference and y are the variables
        """
        n1 = self.n1
        n2 = self.n2
        N = self.N
        M = self.M
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
        """
        This function starts the relaxation algorithm and solves the finite difference equations until converage.
        Convergance is achieved when err < tol or the max number of iterations is achieved. Uses scipy to reduce
        the S matrix.

        Attributes:
        -----------
            errList : list
                Contains the errors at each iteration. updates self.err
            itList : list
                Contains the cummulative count of the iterations. updates self.iterations
            tol : float
                Cut off for convergance

        Returns:
        --------
            x : numpy array
                Mesh grid
            y : numpy array
                The final solutions to the differential equations that respects the boundary conditions
        """
        errList = []
        itList = []
        x = self.x
        y = self.y
        for it in range(self.iteration):
            resList=self.CalcRes(x, y)
            s = self.CalcS(x, y)
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore")
                deltayList = spsolve(s, -resList)
            err = self.CalcError(deltayList)
            if(self.display): print(it,err)
            errList.append(err)
            itList.append(it)
            stopIteration = err < self.tol
            if not stopIteration: deltayList = self.slowc/max([self.slowc,err])*deltayList
            y = self.UpdateY(y, deltayList)
            if stopIteration: break
            if it == self.iteration-1:
                self.successful = False
                break 
        self.err = errList
        self.iterations = itList
        return x, y



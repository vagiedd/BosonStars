# Bose-Einstein Condensate Solver
Numerically solve the Einstein-Klein-Gordon equations of motion for two scalar boson stars, with relaxation method. 

Important files are:

1. relaxation_method.py
2. bosonstar.py
3. parallel.py

Optional files are:

1. main.ipynb
2. test1.py
3. test2.py

The code is written and tested in Python 2.7 and 3.

## Relaxation Method 
Implements the relaxation algorithm given in "Numerical Recipies in C: The Art of Scientific Computing". It contains the relaxation_method class and its initialization function is 

    def __init__(self,xmin,xmax,M,N,
                 B1=[],B2=[],y_guess=[],iteration=100,
                 start=False,display=False,**kwargs):

where 
- **xmin** = initial x value
- **xmax** = final x value 
- **M** = length of x array
- **N** = number of differential equations
- **y_guess** = matrix of initial guesses (or empty) which contains the solution to the differential equations. must have shape N x M
- **B1** = list of initial conditions
- **B2** = list of boundary conditions
- **start** = automatically start relaxation method if True. calls start function
- **display** = display results of each iteration of the relaxation method
- **slowc** = not specified in initialization call but can be changed by self. change it will impact the rate of convergence. 

The relaxation method is started with calling the function

    def start(self):
        ...
        return self.x, self.y

where 
- **self.x** = grid given by np.array(xmin,xmax,M)
- **self.y** = solutions of the differential equations which has shape N x M

Can be called with 

    res = relaxation_method(...,start=False)
    x,y = res.start
    dic = res.dic
or 

    res = relaxation_method(...,start=True)
    x,y = res.x,res.y
    dic = res.dic

The iteration counts, error at each iteration, and if the relaxation method was successfull is contained in the dictionary given by res.dic. 

To implement the relaxation class, you can implement your own extensions of the relaxation_method class. Examples are given in test1.py and test2.py. 

The extension is given by

    class test(relaxation_method):
   
You must define the following function that contains the differential equations with shape N

        def f(self,t,y):
            #y[0] = y
            y = y.item(0)
            f = [y*np.cos(t+y)]
            return np.array(f)
            
You can define the jacobian of the differential equations with shape NxN. If not specified, the algorithm will compute it using finite differences such as in test2.py

        def jac(self,t,y):
            #y[0] = y
            y = y.item(0)
            jac = [[np.cos(t+y) - y*np.sin(t+y)]]
            return np.array(jac)
            
You must specify the jacobian of the derivatives of your boundary conditions with shape n1 x N where n1 is the number of initial boundary conditions. Elements are either 0 or 1 if the boundary condition in that position is specified or not.

        def dB1dy(self):
            #shape n1 x N
            dB1dy = np.zeros((self.n1,self.N))
            dB1dy[0][0] = 1.0
            #dB1dy[1][1] = 1.0
            return dB1dy

        def dB2dy(self):
            #shape n2 x N
            dB2dy = np.zeros((self.n2,self.N))
            #dB2dy[0][0] = 1.0
            return dB2dy

The function scale just specifies the relative scale of each variable in the differential equation.  It can have shape N where k returns the position in the list

        def scale(self,k):  
            return 0.25

For examples see test1.py and test2.py or the Numerical Recipies book for further explanation of the relaxation method. 

## Boson Star 
Implements the extension of the relaxation_method to solve the Einstein-Klein-Gordon equations of motions for a system of 2 scalars in bosonstar.py. It contains the functions necessary for the relaxation method. 

    class bosonstar(relaxation_method):
        def init(self,lam1=1,lam2=1,lam12=1,m1=1e-10,m2=1e-10,fr=1e17):

where 
- **lam1** = coupling of first scalar
- **lam2** = coupling of second scalar
- **lam12** = interaction between two scalars
- **m1** = mass of first scalar
- **m2** = mass of second scalar
- **fr** = decay scale of scalars. 

If fr is set to None, the couplings are the rescaled quantities.  Although the bosonstar class can be called directly, the file contains helper functions to return the solutions to the equation of motion as well as the class self to have access to the bosonstar object created. The functions are 

    def get_profiles(phi10,phi20,
                     lam1=1,lam2=1,lam12=1,
                     m1=1e-10,m2=1e 10,fr=1e17,
                     rstart=10.0,dx=1.0,
                     display=False,iteration=100):
        ...             
        return x,y,res

where 
- **phi10** = initial central density of the first scalar
- **phi20** = intial central density of the second scalar
- **rstart** = initial xmax value to pass to the relaxation method. this will be updated
- **dx** = step size to update xmax
- **display** = display results of relaxation method
- **iteration** = number of allowed iterations

returns 
- **x** = grid
- **y** = solutions of Einstein-Klein-Gordon equations of motion
- **res** = bosonstar object

Can return the total mass, compactness, central radius, and central density of the boson star. 

Just compactness and total mass: 

    def get_compactness(x,y,res):
        Cbs,Mbs,rc,rho = res.findMbs(x,y)
        return Cbs,Mbs
        
inputs are x,y and res returned from get_profiles

returns 
- **Cbs** = compactness (dimensionless)
- **Mbs** = total mass of boson star in solar masses

Just central density and central radius:

    def get_density(x,y,res):
        Cbs,Mbs,rc,rho = res.findMbs(x,y)
        return rc,rho
inputs are x,y and res returned from get_profiles. 
returns
- **rc** = dimensionless central radius
- **rho** = dimensionless central density
 
Similarly if you want Cbs, Mbs, rc, and rho just call res.findMbs(x,y) from the returned object from get_profiles. 

The bosonstar.py file also contains a helper function to evaluate multiple points at the same time in parallel.  

    def scan(phis,cores=1,parallel={"_print":True,"split":True},**kwargs):

Example usage

    kwargs = dict(lam1=1,lam2=1,lam12=1,m1=1e-10,m2=1e-10,fr=1e17)
    phi2 = 10**(np.linspace(np.log10(1e-4),np.log10(8e-2),30))
    phi1 = np.zeros_like(phi2)
    phis = np.column_stack((phi1,phi2))
    results = scan(phis,cores=30,**kwargs)
  
where results is initialized to be 

    results = {"phi1c":[],"phi2c":[],"Cbs":[],"Mbs":[],"M2bs":[],"C2bs":[],"Rc":[],"Rhoc":[]}

and it will have the shape of the input phis which contains the central densities of the scalar fields. 

## Jupyter Notbook: main.ipynb

For examples on calling the functions in bosonstar.py please see the jupyter notebook

## Using the code
You can use this code freely, provided that in your publications, you cite

    @article{Guo:2020tla,
        author = "Guo, Huai-Ke and Sinha, Kuver and Sun, Chen and Swaim, Joshua and Vagie, Daniel",
        title = "{Two-Scalar Bose-Einstein Condensates: From Stars to Galaxies}",
        eprint = "2010.15977",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        month = "10",
        year = "2020"
    }


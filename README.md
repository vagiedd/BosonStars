# BosonStars
Numerically solve the Einstein-Klein-Gordon equations of motion for two scalar boson stars. 

Important files are:

1. relaxation_method.py
2. bosonstar.py
3. parallel.py

Optional files are:

1. main.ipynb
2. test1.py
3. test2.py

## Relaxation Method 
Implements the relaxation algorithm given in "Numerical Recipies in C: The Art of Scientific Computing". It contains the relaxation_method class and its initialization function is 

    def __init__(self,xmin,xmax,M,N,
                 B1=[],B2=[],y_guess=[],iteration=100,
                 start=False,display=False,**kwargs):

where 
- **xmin** = initial x value
-**xmax** = final x value 
-**M** = length of x array
-**N** = number of differential equations
-**y_guess** = matrix of initial guesses (or empty) which contains the solution to the differential equations. must have shape N x M
-**B1** = list of initial conditions
-**B2** = list of boundary conditions
-**start** = automatically start relaxation method if True
-**display** = display results of each iteration of the relaxation method
       

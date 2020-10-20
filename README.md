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

  def __init__(self,xmin,xmax,M,N,B1=[],B2=[],y_guess [],iteration=100,start=False,display=False,**kwargs):
       

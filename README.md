# Using Gaussian Kernel Density Estimation to Simulate Safety Stock

![](https://i.imgur.com/FaWOaYJ.png)

<h2>Description</h2>

The typical formula for safety stock is given as $= Z_\sigma \sigma_d \sqrt{L + R}$	

where $= Z_\sigma$ is the inverse CDF, $\sigma_d$ is the standard deviation of demand, $L + R$ i sthe Lead time and Review period

some notable limitations when applying this formulation to real-world situations are as follows
1. demand is not normally distributed and mostly discrete
2. demand is historical and not forward looking
3. the  $= Z_\sigma$ is dependent on picking an arbitrary service level, but may not be optimal

In this, project, the following modifications were made to the above formulation to minimize the limitations when applying to real-world scenario
1. model demand using a custom gaussian kernel density estimation and transform the resulting distribution to a discrete distribution
2. use forecasted demand if available
3. run simulations over different service levels say 70% to 99% and different review periods say 1 week to 4 weeks
4. optional: use linear programming to pick the safety stock setting that minimizes inventory holding cost


<h2>Programming Language</h2>

- Python

<h2>Environment Used </h2>

- <b>Ubuntu</b>

<h2>To reproduce:</h2>

<p align="left">
 
1. Clone the project: Run this from the command line
 
 ```commandline
 git clone git@github.com:graphshade/zeno_convex.git
 ```
 
2. Change directory to safety_stock_simulation, create an virtual environment and install the dependencies.
   
```commandline
 pip install -r requirements.txt
 ```

3. update the main.py file to input needed data and run the main.py file
```commandline
python main.py
 ``` 




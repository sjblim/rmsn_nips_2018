# Forecasting Treatment Responses Over Time Using Recurrent Marginal Structural Networks
> by Bryan Lim, Ahmed Alaa and Mihaela van der Schaar, **NeurIPS 2018**

## Usage instructions
* To generate a new simulation, run hyperparameter optimisation and tests for the RMSN use : 


    > **bash test_rmsn.sh**
* This produces the raw MSEs into the "*results*" folder, and saves calibrated models into the "*models*" folder
* Note that for the paper, results are imported in normalised root mean-squared error terms -> i.e. sqrt(mse) / 1150

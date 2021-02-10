
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#ex1">Ex1 evaluate/plot bivariate Gaussian PDF</a>
    <li>
      <a href="#ex2">Ex2 analyse sea level pressure (SLP) anomalies</a>
    <li><a href="#procedure">Procedure</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>


## Ex1 
TASK: evaluate/plot bivariate Gaussian PDF
#### FILE: ex1/ex1_part_1.py 
Input: 2 mean values, 2 variances, correlation parameter.
- PDF class calculates bivariate Gaussian PDF values at each location in the 2D grid. 
- Create covariances 2x2 covariance matrix using input variances and covariances, calculated using correlation parameter
- create mehsgrid to be filled with PDF values
- calculate PDF value for each meshgrid cell using Gaussian PDF equation in matrix form
- plot PDF

#### FILE:  ex1/ex1_part_2.py 
Input: 2 mean values, 2 variances, correlation parameter, sample size of RV -m. 
- Create covariances 2x2 covariance matrix using input variances and covariances, calculated using correlation parameter
- Get eigenvalues, eigenvectors of covariance matrix
- Get  2 × m matrix A, drawing m random samples from a normal (Gaussian) distribution
- Generate matrix Mu , containing two arrays of mean values of shape corresponding to
      size of  arrays of independent random samples (m x m)
 - Get X (2 x m)  as X=np.dot(E,A)+Mu
 - plot generated PDF for different correlation parameters

## Ex2 
TASK: analyse sea level pressure (SLP) anomalies
#### FILE:  ex2/ex2_part1_Gordeev.py 
Input: HadSLP2 netcdf, 2 variances, correlation parameter.
- Extract December months from netcdf file using xarray, plot the longterm mean SLP field for December 
- Extract SLP time series from Reykjavik/Lisbon grid cells
- Plot 2D histogram with Reykjavik/Lisbon time series SLP data on x and y axis
- calulate bivariate normal distribution parameters(mean, variance
and correlation) from Reykjavik/Lisbon time series
- generate and plot PDF with estimated parameters (same way as in Ex1)



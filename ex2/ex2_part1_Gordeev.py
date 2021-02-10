import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numpy.linalg as LA
from cdo import *
import os

class PDF:
    def __init__(self,means,variances,covmat):
        self.means = means
        self.variances = variances
        self.covmat = covmat


    def get_xvect(self,mean,sigma):
        # input X vector
        xstart=mean-3*np.sqrt(sigma)
        xend=mean+3*np.sqrt(sigma)
        xvect=np.arange(xstart,xend,1)
        return xvect


    def get_PDF_value(self,xi):
        covmat=self.covmat
        covmat_inv=LA.inv(covmat)
        diffvect=xi-self.means
        D=np.dot(np.dot(diffvect,covmat_inv),diffvect)
        D
        PDF_value=np.exp(-D)/(2*np.pi*np.sqrt(LA.det(covmat)))

        return PDF_value

    def get_meshgrid(self):
        x1=self.get_xvect(means[0],variances[0])
        x2=self.get_xvect(means[1],variances[1])
        meshgrid=np.zeros([len(x1),len(x2),2])
        for i in range(len(x1)):
            for j in range(len(x2)):
                meshgrid[i,j]=tuple([x1[i],x2[j]])

        return meshgrid

    def calc_PDF_arr(self):
        meshgrid=self.get_meshgrid()
        len_x1,len_x2=meshgrid.shape[:-1]
        cdf_arr=np.zeros([len_x1,len_x2])
        for i in range(len_x1):
            for j in range(len_x2):
                cdf_arr[i,j]=self.get_PDF_value(meshgrid[i,j])

        return cdf_arr

    def plot_PDF_arr(self):
        cdf_arr=self.calc_PDF_arr()
        fig=plt.figure(figsize=(12,10))
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.pcolor(cdf_arr)
        plt.title('Joint PDF',fontsize=20)
        plt.colorbar()
        plt.show()


cdo = Cdo()
selected_months='/home/igor/UNI/KLIMDYN/Hometask/ex2/DEC.nc'
infile='/home/igor/UNI/KLIMDYN/Hometask/ex2/HadSLP2_slp-mnmean-real.nc'
cdo.selmon(12,input = infile,output = selected_months)

ds = xr.open_dataset(selected_months)
SLP_dec=ds.slp
SLP_dec.shape
SLP_dec[:,12,5]



# calculating mean along time axis
SLP_dec_mean=SLP_dec.mean(axis=0)
# Plot the longterm mean SLP field for December
xlen=len(SLP_dec_mean.lon)
ylen=len(SLP_dec_mean.lat)
fig=plt.figure(figsize=(18,9))
plt.imshow(SLP_dec_mean)
plt.yticks([i for i in range(0,ylen,2)],SLP_dec_mean.lat.values[::2])
plt.xticks([i for i in range(0,xlen,4)],SLP_dec_mean.lon.values[::4])
plt.title('SLP december mean (1850-2019)',fontsize=16)
plt.xlabel('longitude')
plt.ylabel('lattitude')
plt.colorbar()
plt.show()


# Assign Rejkavik and Lisbon grid cell indices
Rx_ind,Ry_ind=5,4
Lx_ind,Ly_ind=2,10
RSLP=SLP_dec[:,Ry_ind,Rx_ind].values
LSLP=SLP_dec[:,Ly_ind,Lx_ind].values
# Plot a 2D-histogram
fig=plt.figure(figsize=(12,10))
plt.hist2d(LSLP,RSLP,bins=15)
plt.title('2 histogram',fontsize=20)
plt.xlabel('Lisbon SLP',fontsize=14)
plt.ylabel('Rejkjavik SLP',fontsize=14)
plt.colorbar()
plt.show()

#calulate bivariate normal distribution parameters
RSLP_mean=RSLP.mean()
LSLP_mean=LSLP.mean()
LSLP_variance=np.mean((LSLP-LSLP_mean)**2)
RSLP_variance=np.mean((RSLP-RSLP_mean)**2)
RSLP_covariance=np.mean((RSLP-RSLP_mean)*(LSLP-LSLP_mean))
means=np.array([RSLP_mean,LSLP_mean])
variances=np.array([RSLP_variance,LSLP_variance])
covmat=np.array([[RSLP_variance,RSLP_covariance],[RSLP_covariance,LSLP_variance]])
covmat

PDF_obj1=PDF(means,variances,covmat)
means[0]

PDF_obj1.plot_PDF_arr()

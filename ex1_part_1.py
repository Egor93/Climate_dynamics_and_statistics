
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt


class CDF:
    def __init__(self,means,variances,correlation,x1_vect,x2_vect):
        self.means = means
        self.variances = variances
        self.correlation = correlation
        self.x1_vect = x1_vect
        self.x2_vect = x2_vect

    def get_covmat(self):
        variance1=self.variances[0]
        variance2=self.variances[1]
        covariance12=self.correlation*np.sqrt(variance1)*np.sqrt(variance2)
        covmat=np.array([[variance1,covariance12],[covariance12,variance2]])

        return covmat

    def get_CDF_value(self,xi):
        covmat=self.get_covmat()
        covmat_inv=LA.inv(covmat)
        diffvect=xi-self.means
        D=np.dot(np.dot(diffvect,covmat_inv),diffvect)
        D
        CDF_value=np.exp(-D)/(2*np.pi*np.sqrt(LA.det(covmat)))

        return CDF_value

    def get_meshgrid(self):
        x1=self.x1_vect
        x2=self.x2_vect
        meshgrid=np.zeros([len(x1),len(x2),2])
        for i in range(len(x1)):
            for j in range(len(x2)):
                meshgrid[i,j]=tuple([x1[i],x2[j]])

        return meshgrid

    def calc_CDF_arr(self):
        meshgrid=self.get_meshgrid()
        len_x1,len_x2=meshgrid.shape[:-1]
        cdf_arr=np.zeros([len_x1,len_x2])
        for i in range(len(x1)):
            for j in range(len(x2)):
                cdf_arr[i,j]=self.get_CDF_value(meshgrid[i,j])

        return cdf_arr

    def plot_CDF_arr(self):
        cdf_arr=self.calc_CDF_arr()
        fig=plt.figure(figsize=(12,10))
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.pcolor(cdf_arr)
        plt.title('Joint PDF, correlation={}'.format(self.correlation),fontsize=20)
        plt.colorbar()
        plt.show()


# INPUT VARAIABLES
# mean array
mu1=50
mu2=50
means=np.array([mu1,mu2])
# variances array
var1=500
var2=500
variances=np.array([var1,var2])
# correlation [-1;1]
corr=0.5
# input X vector
xstart=1
xend=100
x1=np.arange(xstart,xend,1)
x2=np.arange(xstart,xend,1)
CDF_obj1=CDF(means,variances,corr,x1,x2)
CDF_obj1.get_covmat()
CDF_obj1.plot_CDF_arr()
# change correlation to -0.5
corr2=-0.5
CDF_obj2=CDF(means,variances,corr2,x1,x2)
CDF_obj2.get_covmat()
CDF_obj2.plot_CDF_arr()
# change correlation to 0
corr3=0.0
CDF_obj3=CDF(means,variances,corr3,x1,x2)
CDF_obj3.get_covmat()
CDF_obj3.plot_CDF_arr()
# CONCLUSION
# the maps looks exactly how I would expect them to see, considering mean and correlation values

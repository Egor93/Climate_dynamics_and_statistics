
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt


class PDF:
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

    def get_PDF_value(self,xi):
        covmat=self.get_covmat()
        covmat_inv=LA.inv(covmat)
        diffvect=xi-self.means
        D=np.dot(np.dot(diffvect,covmat_inv),diffvect)
        PDF_value=np.exp(-D)/(2*np.pi*np.sqrt(LA.det(covmat)))

        return PDF_value

    def get_meshgrid(self):
        x1=self.x1_vect
        x2=self.x2_vect
        meshgrid=np.zeros([len(x1),len(x2),2])
        for i in range(len(x1)):
            for j in range(len(x2)):
                meshgrid[i,j]=tuple([x1[i],x2[j]])

        return meshgrid

    def calc_PDF_arr(self):
        meshgrid=self.get_meshgrid()
        len_x1,len_x2=meshgrid.shape[:-1]
        cdf_arr=np.zeros([len_x1,len_x2])
        for i in range(len(x1)):
            for j in range(len(x2)):
                cdf_arr[i,j]=self.get_PDF_value(meshgrid[i,j])

        return cdf_arr

    def plot_PDF_arr(self):
        cdf_arr=self.calc_PDF_arr()
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
PDF_obj1=PDF(means,variances,corr,x1,x2)
PDF_obj1.get_covmat()
PDF_obj1.plot_PDF_arr()
# change correlation to -0.5
corr2=-0.5
PDF_obj2=PDF(means,variances,corr2,x1,x2)
PDF_obj2.get_covmat()
PDF_obj2.plot_PDF_arr()
# change correlation to 0
corr3=0.0
PDF_obj3=PDF(means,variances,corr3,x1,x2)
PDF_obj3.get_covmat()
PDF_obj3.plot_PDF_arr()
# CONCLUSION
# the maps looks exactly how I would expect them to see, considering mean and correlation values


import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt


class PDF:
    # DESCRIPTION
	# 		Class containing all methods necessary to generate
    #       two arrays of samples which have user-defined standart deviation,mean;
    #       correlate/do not correlate with each other according to user input.
	# INPUT
	#		means
    #       variances
    #       correlation   -    correlation coefficient (of two sample arrays)
    #       a_size        -    size of  arrays of independent random samples
	# OUTPUT
	#      plot_PDF_hist method, plots the 2D histogram of generated samples
    def __init__(self,means,variances,correlation,a_size):
        self.means = means
        self.variances = variances
        self.correlation = correlation
        self.a_size = a_size

    def get_covmat(self):
        # DESCRIPTION
    	# 		Get covariance matrix using informaion
        #       about variances and correlation
        variance1=self.variances[0]
        variance2=self.variances[1]
        covariance12=self.correlation*np.sqrt(variance1)*np.sqrt(variance2)
        covmat=np.array([[variance1,covariance12],[covariance12,variance2]])

        return covmat

    def get_eigen(self):
        # DESCRIPTION
    	# 		Get eigenvalues,eigentvectors of covariance matrix
        covmat=self.get_covmat()
        eigvals,eigvects=LA.eig(covmat)

        return eigvals,eigvects

    def get_A(self):
        # DESCRIPTION
    	# 		Generate matrix A, containing two
        #       arrays of samples.
        eigvals,eigvects=self.get_eigen()
        variances=eigvals
        # these standart deviations coincide with input stdiv
        # if correlation=0, otherwise there is some deviation from input stdiv.
        stdeviations=np.sqrt(variances)
        # Draw m random samples from a normal (Gaussian) distribution.
        a1=np.random.normal(0,stdeviations[0],self.a_size)
        a2=np.random.normal(0,stdeviations[1],self.a_size)
        A=np.array([a1,a2])

        return A

    def get_E(self):
        # DESCRIPTION
    	# 		Generate matrix E,using
        #       eigenvectors of covariance matrix.
        return self.get_eigen()[1]

    def get_Mu(self):
        # DESCRIPTION
    	# 		Generate matrix Mu, containing two
        #       arrays of mean values of shape corresponding to
        #        size of  arrays of independent random samples - a_size
        mu1=np.full((1,self.a_size),means[0])
        mu2=np.full((1,self.a_size),means[1])
        Mu=np.array([mu1[0],mu2[0]])

        return Mu

    def calc_X(self):
        E = self.get_E()
        A = self.get_A()
        Mu = self.get_Mu()

        X=np.dot(E,A)+Mu

        return X

    def plot_PDF_hist(self):
        X=self.calc_X()
        fig=plt.figure(figsize=(12,10))
        plt.hist2d(X[0],X[1],bins=int(self.a_size/10))
        plt.title('Joint PDF/2 random variables/{} samples, correlation={}'.format(self.a_size,self.correlation),fontsize=20)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.colorbar()
        plt.show()


# INPUT VARAIABLES
# mean array
mu1=50
mu2=40
means=np.array([mu1,mu2])
# variances array
var1=400
var2=500
variances=np.array([var1,var2])

# correlation [-1;1]
corr1=0.5
# size of vector a (m elements)
a_size=1000
PDF_obj1=PDF(means,variances,corr1,a_size)
PDF_obj1.plot_PDF_hist()

# change correlation to -0.5
corr2=-0.5
PDF_obj2=PDF(means,variances,corr2,a_size)
PDF_obj2.plot_PDF_hist()

# change correlation to 0
corr3=0.0
PDF_obj3=PDF(means,variances,corr3,a_size)
PDF_obj3.plot_PDF_hist()
# CONCLUSION
# the maps looks exactly how I would expect them to see, considering mean and correlation values


import numpy.linalg as LA
import numpy as np

def get_CDF(xi,means,covmat):

    covmat_inv=LA.inv(covmat)
    diffvect=xi-means
    D=np.dot(np.dot(diffvect,covmat_inv),diffvect)
    D
    CDF=np.exp(-D)/(2*np.pi*np.sqrt(LA.det(covmat)))

    return CDF


# covar matrix
covmat=np.array([[1,0],[0,1]])
# mean array
mu1=20
mu2=30
means=np.array([mu1,mu2])
# input X vector
xstart=1
xend=100
x1=np.arange(xstart,xend,1)
x2=np.arange(xstart,xend,1)
xvect=[[x1[i],x2[i]] for i in range(len(x1))]
xvect
covmat

arr=[0]
arr[0]=[1]
arr

meshgrid=np.zeros([len(x1),len(x2),2])
meshgrid.shape
meshgrid[1,1]=[1,2]
for i in range(len(x1)):
    for j in range(len(x2)):
        meshgrid[i,j]=tuple([x1[i],x2[j]])


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


Y1 = np.array([42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15])
Y = np.matrix(Y1).T
X1 = np.array([[80, 27, 89],
           [80, 27, 88],
           [75, 25, 90],
           [62, 24, 87],
           [62, 22, 87],
           [62, 23, 87],
           [62, 24, 93],
           [62, 24, 93],
           [58, 23, 87],
           [58, 18, 80],
           [58, 18, 89],
           [58, 17, 88],
           [58, 18, 82],
           [58, 19, 93],
           [50, 18, 89],
           [50, 18, 86],
           [50, 19, 72],
           [50, 19, 79],
           [50, 20, 80],
           [56, 20, 82],
           [70, 20, 91]])

np.random.seed(1)
T = Y.shape[0]
#for j in range(X1.shape[1]):
#    X1[:,j] = (X1[:,j] - np.mean(X1[:,j]))/np.std(X1[:,j])
X = np.c_[np.ones([T,1]),X1]
K = X.shape[1]

#step 1 set priors and starting values
# Normal prior for B
B0 = np.array([[10],[0],[0],[0]])  #prior mean
Sigma0 = np.sqrt(1/0.00001)*np.eye(K)  #prior variance
#Inverse gamma priors for sigma2 i.e the noise variance
T0 = 10**(-2)  #prior degrees of freedom for inverse gamma distribution
D0 = 10**(2) #prior scale parameter
#(shape, rate) = 10**(-2) rate = 1/scale

#starting values
B = B0
sigma_sq = 10 #tau = 0.1 # noise variance

reps=11000   #total numbers of Gibbs iterations
burn=1000   #percent of burn-in iterations
out1=np.zeros([4,1])
out2=[]
for i in range(reps):
    #step 2 Sample B conditional on sigma N(M*,V*)
    M=np.dot(inv(inv(Sigma0)+(1/sigma_sq)*np.dot(X.T,X)),(np.dot(inv(Sigma0),B0)+(1/sigma_sq)*np.dot(X.T,Y)))
    V=inv(inv(Sigma0)+(1/sigma_sq)*np.dot(X.T,X))
    
    B = np.random.multivariate_normal(np.asarray(M.T)[0],V)
    B = B.reshape([K,1])
    #step 3 sample sigma2 conditional on B from IG(T1,D1);
    #compute residuals
    resids=Y-np.dot(X,B)
    #compute posterior df and scale matrix
    T1=T0+T;
    D1=D0+np.dot(resids.T,resids)
    #draw from IG
    z0 = np.random.normal(0,1,(T1,1))
    z0z0=np.dot(z0.T,z0)
    sigma_sq=D1/z0z0
    sigma_sq = sigma_sq[0,0]

    if i>burn:
        out1= np.c_[out1, B.T[0]]
        out2= np.r_[out2,sigma_sq]

coeff = np.mean(out1, axis =1).reshape([K,1])
pred1 = np.dot(X,B)
plt.figure()
plt.plot(pred1, 'r', label ='Predicted')
plt.plot(Y, 'g', label = 'Actual')
plt.legend()

rmse = np.sqrt(np.sum(np.square(Y-pred))/T)
print ('rmse is:', rmse)

plt.figure()
plt.plot(pred1, 'r', label ='MCMC')
plt.plot(pred, 'y', label ='VI')
plt.plot(Y, 'g', label = 'Actual')
plt.legend()




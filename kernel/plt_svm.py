import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def kernel_poly2(X,Y):
    X = np.concatenate([X,(X[:,0]*X[:,1]).reshape(-1,1)], axis=1)
    Y = np.concatenate([Y,(Y[:,0]*Y[:,1]).reshape(-1,1)], axis=1)
    return np.dot(X,Y.T)

# Data
c = 0.5
np.random.seed(0)
N = 200
X = np.concatenate([
    c*np.random.randn(N//4, 2)+[1,1],
    c*np.random.randn(N//4, 2)+[1,-1],
    c*np.random.randn(N//4, 2)+[-1,1],
    c*np.random.randn(N//4, 2)+[-1,-1],
], axis=0)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)


# Plot
height = 5
fig, axs = plt.subplots(1, 5, figsize=(height*5,height))
font = {'fontsize': 50, 'fontname': 'Times New Roman'}
cmap = plt.cm.get_cmap('bwr')

# Decision function 
def plt_decision_function(ax, kernel, gamma=1):
    # gamma = 1/\sigma^2
    clf = svm.SVC(kernel=kernel, gamma=gamma)
    clf.fit(X, Y)
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 500),
                        np.linspace(-2.5, 2.5, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    nlevel = 3
    levels = np.linspace(-1,1,nlevel)
    linestyles = ['--' for _ in range(len(levels))]
    linestyles[nlevel//2] = '-'
    ax.contour(xx, yy, Z, levels=levels, linewidths=2,
               linestyles=linestyles, colors=[cmap(0.2),cmap(0.1),cmap(0.2)])
    
    y = np.sign(clf.decision_function(X))
    mask0 = np.ma.masked_where(y==-1, [30 for _ in range(N)])
    mask1 = np.ma.masked_where(y==1,  [30 for _ in range(N)])
    ax.scatter(X[:,0], X[:,1], s=mask0, c=[cmap(0.9)], marker='x')
    ax.scatter(X[:,0], X[:,1], s=mask1, c=[cmap(0.6)], marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    if clf.support_vectors_.shape[0] != 0:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                       linewidth=1, facecolors='none', edgecolors=cmap(0.95))
    return ax
        
        

# scatter
ax = axs[0]
mask0 = np.ma.masked_where(Y==0, [30 for _ in range(N)])
mask1 = np.ma.masked_where(Y==1,  [30 for _ in range(N)])
ax.scatter(X[:,0], X[:,1], s=mask0, c=[cmap(0.9)], marker='x')
ax.scatter(X[:,0], X[:,1], s=mask1, c=[cmap(0.6)], marker='o')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Data', **font)

# RBF kernel
for i, gamma in enumerate([4,1,0.25]):
    ax = axs[1+i]
    ax = plt_decision_function(ax, 'rbf', gamma)
    ax.set_title(f'rbf ($\sigma^2={1/gamma}$)', **font)

# poly kernel
ax = axs[4]
ax = plt_decision_function(ax, kernel_poly2)
ax.set_title(f'polynomial$^2$', **font)

plt.tight_layout()
save_path = './summary/assets/svm_on_2d_gaussian_vary_kernel'
fig.savefig(save_path, bbox_inches='tight', dpi=100)

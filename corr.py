

import numpy as np
from sklearn.cross_decomposition import CCA

X = np.array([[1,0,-1,0],[2,0,-2,0],[1,0,-2,0]]).T

Y = np.array([[2,1,-1,0],[0,1,0,-1]]).T

cca = CCA(n_components=1)
cca.fit(X, Y)

x_c, y_c = cca.transform(X, Y)

print(f"x_c = {x_c}, y_c = {y_c}")

corr = np.corrcoef(x_c.T, y_c.T)[0, 1]
print("Canonical correlation =", corr)

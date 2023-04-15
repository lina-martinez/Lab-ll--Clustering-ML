from TSNE import TSNE as MyTSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load DataSet
wine_data = load_wine()
X, y = wine_data['data'], wine_data['target']

X_scaled = StandardScaler().fit_transform(X)

# Create a PCA robust object with 2 components
PCArobust = MyTSNE(n_components=2)
# Fit the data and transform
PCArobust_fit = PCArobust.fit(X_scaled)
PCArobust_transform = PCArobust.transform(X_scaled)

print(PCArobust_transform.shape)

plot = plt.scatter(PCArobust_transform[:,0], PCArobust_transform[:,1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=list(wine_data['target_names']))
plt.show()
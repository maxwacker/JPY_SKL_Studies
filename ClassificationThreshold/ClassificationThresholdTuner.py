# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# From:  https://scikit-learn.org/stable/modules/classification_threshold.html

# %%
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

Nobs = 100
Nfeats = 2

X, y = make_classification(n_samples= Nobs, n_features=Nfeats, n_redundant=0, random_state=0)
classifier = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
classifier.predict_proba(X[:4])
classifier.predict(X[:4])

# %%
X.shape # -> (obs, vars)  

# %%
X[0,:]

# %%
cmp = np.array(['r', 'g', 'b'])

fig = plt.figure(figsize=(12, 9))
plt.scatter(X[:,0], X[:,1], c=cmp[y], s=50, edgecolors='none')
plt.show()


# %%

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

Nobs = 64 #16
Nfeats = 2

X, y = make_classification(n_samples= Nobs, n_features=Nfeats, n_redundant=0, random_state=0)
#classifier = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
#classifier.predict_proba(X[:4])
#classifier.predict(X[:4])

# %%
X.shape # -> (obs, vars)  

# %%
# Spliting Train/Test & Plotting 
from sklearn.model_selection import train_test_split

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:,0],X_train[:,1],c=cmp[y_train], s=50, edgecolors='none', label="Train")
plt.scatter(X_test[:,0], X_test[:,1], c='none' ,s=50, edgecolors=cmp[y_test], label="Test")
#plt.legend(handles=[train_handle, test_handle])
leg = plt.legend()
leg.legend_handles[0].set_color('black')      # Train
leg.legend_handles[1].set_facecolor('none')   # Test - already empty, just fix edge
leg.legend_handles[1].set_edgecolor('black')
plt.show()


# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
# évaluation et affichage sur split1
lda.fit(X_train, y_train)
print("Le score sur le jeu d'apprentissage est de : {:.3f}".format(lda.score(X_train, y_train)))

print("Le score sur le jeu de test est de : {:.3f}".format(lda.score(X_test, y_test)))

# %%
# on créé une nouvelle figure sur laquelle on affiche les points
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:,0], X_train[:,1], c=cmp[y_train], s=50, edgecolors='none', label="Train")
plt.scatter(X_test[:,0],  X_test[:,1], c='none', s=50, edgecolors=cmp[y_test], label="Test")

# on calcule pour chaque point du plan sa probabilité d'appartenir à chaque classe
nx, ny = 400, 400
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
# meshgrid permet d'échantillonner tous les points du plan (entre x_min et x_max)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
# .predict_proba permet de prédire le score de la LDA pour un ensemble d'observations
Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
for cls_idx in range(max(y)):
    zz = Z[:, cls_idx].reshape(xx.shape)
    # on dessine la frontière correspond à un score de 0,5
    # les scores < 0,5 correspondent à la classe 0
    # les scores > 0,5 correspondent à la classe 1
    plt.contour(xx, yy, zz, [0.5])
leg = plt.legend()
leg.legend_handles[0].set_color('black')      # Train
leg.legend_handles[1].set_facecolor('none')   # Test - already empty, just fix edge
leg.legend_handles[1].set_edgecolor('black')
plt.show()

# %%
# Confusion Matrix 
# from: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions = lda.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=lda.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lda.classes_)
disp.plot()
plt.show()

# %%
np.stack(( y_test, predictions), axis=1).shape#[:4,:]


# %%
# Qualifying Predictions
def true_positives(actuals, predictions):
    actuals_predictions = np.stack(( actuals, predictions), axis=1)
    return int(np.sum(np.all(actuals_predictions == [1, 1], axis=1)))

def true_negatives(actuals, predictions):
    actuals_predictions = np.stack(( actuals, predictions), axis=1)
    return int(np.sum(np.all(actuals_predictions == [0, 0], axis=1)))

def false_positives(actuals, predictions):
    actuals_predictions = np.stack(( actuals, predictions), axis=1)
    return int(np.sum(np.all(actuals_predictions == [0, 1], axis=1)))

def false_negatives(actuals, predictions):
    actuals_predictions = np.stack(( actuals, predictions), axis=1)
    return int(np.sum(np.all(actuals_predictions == [1, 0], axis=1)))


# %%
# Commenting Confusion Matrix Result:
# As seen at begining of NoteBook 
# TRUE class is GREEN  - (cmp[0] is red, cmp[1] is green)

print(f"true_positives ({true_positives(y_test, predictions)}) : GREEN (empty) circle,\ton RIGHT side of plot,\tDOWN-RIGHT in Confusion Matrix")
print(f"true_negatives ({true_negatives(y_test, predictions)}) : RED (empty) circle,\ton RIGHT side of plot,\tUP-LEFT in Confusion Matrix")
print(f"false_positives ({false_positives(y_test, predictions)}) : RED (empty) circle,\ton WRONG side of plot,\tUP-RIGHT in Confusion Matrix")
print(f"false_negatives ({false_negatives(y_test, predictions)}) : GREEN (empty) circle,\ton WRONG side of plot,\tDOWN-LEFT in Confusion Matrix")


# %%

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
# Classification Metrices explained From 
# https://medium.com/data-science/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019
# DataSet From : https://github.com/andrewwlong/classification_metrics_sklearn/blob/master/data.csv

# %%
import pandas as pd
df = pd.read_csv('data.csv')
df.head()

# %%
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

# %%
# My Own practice : Scatter Plot the dataset
import numpy as np
import matplotlib.pyplot as plt

colors = np.where(df['actual_label'], 'r', 'k')
plt.scatter(df.model_RF, df.model_LR, s=10, c=colors)

# Using Pandas plot

#df.plot.scatter('model_RF', 'model_LR', c=colors)

# %%
# Align futur markdown table to the let
# #%%html <style> table {float:left} </style>

# %% [markdown]
#
# ## Lexique
#
# En classication binaire (avec une classe d'intérêt) On a 2 labels binaires : 
# - *actual_label* : le vrai, celui de la supervision
# - *predicted_label* : celui prédit par le modèle considéré
#
# On classe les échantillons en 4 catégories :
# | Sign | Term        | Meaning                  |
# ---|-----------------|--------------------------
# |TP| True Positive   | prédit 1  et  1 "en vrai"|
# |FP| False Postitive | prédit 1 mais 0 "en vrai"|
# |TN| True Negative   | prédit 0  et  0 "en vrai"|
# |FN| False Negative  | predit 0 mais 1 "en vrai"|
#
# ![title](Précision_et_rappel.jpg)
#
#
# ### Résumé (mnémo-technique) :
# - *Le premier terme qualifie la prédition bonne/mauvaise.* 
# - *Le second terme est la valeur prédite (à tort ou à raison)*
#
# Fasle Negative : Faussement prédit Négatif (Positif en vrai)
#
#

# %%
# Matrice de confusion sur notre data set
# 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df.actual_label.values, df.predicted_RF.values)
cm

# %% [raw]
# Interpretation :
#
#             Predicted
#              Neg.  Pos.
#  Actual Neg.  TN   FP
#         Pos.  FN   TP

# %%
# So here we got
cm
# 5519 True Negatives
# 2360 False Positives
# 2832 False Negatives
# 5047 True Positives

# %%
# Confusion Matrix build 'by hand'
def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))# your code here
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))# your code here
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))# your code here


# %%
def find_conf_matrix_values(y_true,y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true,y_pred)
    FN = find_FN(y_true,y_pred)
    FP = find_FP(y_true,y_pred)
    TN = find_TN(y_true,y_pred)
    return TP,FN,FP,TN
def my_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])



# %%
my_confusion_matrix(df.actual_label.values, df.predicted_RF.values)

# %%
assert  np.array_equal(my_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(df.actual_label.values, df.predicted_RF.values) ), 'my_confusion_matrix() is not correct for RF'
assert  np.array_equal(my_confusion_matrix(df.actual_label.values, df.predicted_LR.values),confusion_matrix(df.actual_label.values, df.predicted_LR.values) ), 'my_confusion_matrix() is not correct for LR'

# %% [markdown]
# ## Accuracy Score
# EN : Proportion of correct predictions (over all prediction)
#
# FR : Taux de prédiction correctes
#
# Accuracy_Score = TP+TN / TP+TN+FP+FN
#

# %%
from sklearn.metrics import accuracy_score
accuracy_score(df.actual_label.values, df.predicted_RF.values)


# %%
def my_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)  
    return (TP + TN) / ( TP + TN + FP + FN)

my_accuracy_score(df.actual_label.values, df.predicted_RF.values)


# %%
def my_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)  
    return (TP + TN) / ( TP + TN + FP + FN)
assert my_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert my_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Accuracy RF: %.3f'%(my_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f'%(my_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

# %% [markdown]
# FR : L'accuracy n'est pas une bonne métrique quand la classe d'intétêt est représentée. Par exemple supposons que nous ayant un dataset de diagnostique de maladie, où seulement 1% des individus sont malades. Un modèle de prédiction qui répondrait 'non-malade' pour tous tests (sans même lire les variables) aurraut un taux Accuracy de 99% ! 
# Mais en fait avec ce modèle on détecte 0% des cas qui nous intéressent vraiement : les malades 

# %% [markdown]
# ## Recall Score
# Ce taux de "0% des malades détectés" correspond à une autre métrique qu'on *rappel* (ou *Sensibilité*) ou *recall* (Ou *Sensibilité) 
#
# EN : Proportion of posisitives predicted correctly (over all terrain-posives
#
# FR : Taux de prédiction positives correctes (sur tous les positifs terrains)
#
# Recall_Score = (TP) / (TP + FN)

# %%
from sklearn.metrics import recall_score
recall_score(df.actual_label.values, df.predicted_RF.values)


# %%
def my_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)  
    return (TP) / (TP + FN)

my_recall_score(df.actual_label.values, df.predicted_RF.values)

# %% [markdown]
# Une façon d'augmenter le recall_score d'un modèle est de faire baisser le seuil de détection pour détecter 'plus vite' les postifs.
# Mais en faisant cela, on rique aussi d'augment le nombre de faux positifs.
#
# Une autre métrique, Prescision_Score, mesure cette effet

# %% [markdown]
# ## Precision Score
#
# EN : Proportion of predicted positives that are actually positives (among all positives detections)
#
# FR : Proportion (Taux) de prédictions positives qui sont vraiment (terrain) des positives (parmi toutes les détections positives)
#
# (TP) / (TP + FP) 

# %%
from sklearn.metrics import precision_score
precision_score(df.actual_label.values, df.predicted_RF.values)


# %%
def my_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)  
    return (TP) / (TP + FP)

my_precision_score(df.actual_label.values, df.predicted_RF.values)


# %% [markdown]
# ## F1 Score
#
# EN : Harmonic mean of recall and precision.
# FR : Moyenne harmonique du rappel et de la précsion
#
# F1_Score = 2 * recall * precision / (precision + recall)
#
# Score + haut pour les meilleurs modèles 
#

# %%

# %%
def my_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = my_recall_score(y_true,y_pred)  
    precision = my_precision_score(y_true,y_pred)  
    return 2 * recall * precision / (precision + recall)

my_f1_score(df.actual_label.values, df.predicted_RF.values)

# %%
from sklearn.metrics import f1_score
f1_score(df.actual_label.values, df.predicted_RF.values)


# %%

# %%
def my_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = my_recall_score(y_true,y_pred)  
    precision = my_precision_score(y_true,y_pred)  
    return  2 * recall * precision / (precision + recall)

my_f1_score(df.actual_label.values, df.predicted_RF.values)

# %%

# %%

# %% [markdown]
# Remarquons que nos colonnes de prédictions ont un seuil de détections de 0.5 sur le variable respectives.
# Voici une comparaison des métriques de performance avec un seuil de décision à 0.25

# %%
print('scores with threshold = 0.5')
print('Accuracy RF: %.3f'%(my_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(my_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(my_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(my_f1_score(df.actual_label.values, df.predicted_RF.values)))
print(' ')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(my_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(my_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(my_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(my_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

# %% [markdown]
# Comment déterminer le bon (meilleur) seuil ?
# C'est là qu'intervient la courbe ROC

# %% [markdown]
# La courbe ROC est un outils qui permet d'établir la balance entre le taux de vrai-positifs et faux-poisitifs.
# SciKitLearn fournit 2 fonctions pour calculer la courbe ROC (roc_curve et roc_auc_score). 
# Les entrées de ces fonctions sont :
# - les lables de supervision (actuals)
# - La probalité de prédiction (attention : ce n'est pas le label prédit)
#
# Ces 2 fonctions sont un peu trop complexes pour qu'on les redéfinisse ici par notre propre code.
# On va juste les étudier.
#

# %%
from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

# %% [markdown]
# La fonction roc_curve renvoie 3 listes :
# - thresholds : liste des valeurs de seuil évaluées ordre décroissant (ie :  probabilit de prédiction)
# - fpr : FP rate = taux de faux-positifs  (pour la valeur de seuil associée - de même rang dans la liste)
# - tpr : TP rate = taux de vrais-positifs (pour la valeur de seuil associée - de même rang dans la liste)

# %%
from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)


# %%
import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# %% [markdown]
# **Remarque :**
#
# Un seuil haut correspond à un point de courbe en bas à gauche de la courbe
# Si on demande une forte probalité pour accepté un candidat dans la classe, 
# on devient donc plus stricte et on réduit donc le nombre de positifs (aussi vrai posititifs que faux positifs)
#
# Récipropquement, un seuil bas augmente le nombre de posifis
#
# <MAIS que dire en ce qui concerne les taux relatifs TPR/FPR ?> 
# <Pas si claire>

# %%

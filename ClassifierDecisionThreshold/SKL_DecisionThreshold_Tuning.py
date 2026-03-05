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

# %% [markdown]
# La classification se compose de 2 étapes :
# - L'apprentissage statisitique d'un modèle qui, idéalement, fournit une probabilité P(y|X) - Probabilité de l'observation X d'appartenir à la classe y
# - La prise de décision contrète de classement, construite à partir du modèle
#
# Illustration météorologique :
# - La première phase répond à la question "Quelle sont les chances qu'il pleuve demain"
# - La seconde phase indique s'il faudra prendre son parapluie

# %% [markdown]
# Dans SKL, l'aspect probabiliste est pris en compte par les fonction **predict_proba** qui renvoie pour chaque classe : P(y|X)
# (ou decision_function, pour les classifiers à score de confiance, voir plus bas)
# La décision de choix de lable (de classe) est établie (à partir des infos de probabilité) par la méthode **predict**
#

# %% [markdown]
# En classification binaire, une règle de décision est définie en fonction d'un seuil sur les scores, 
# aboutissant à une prévision d'un label unique pour chaque observation.
#
# Pour une classification binaire donc, les labels de prédiction sont établis par un seuil strict : la classe est déclarée positive quand quand la probabilité conditionnelle P(y|X) est supérieur à 0.5 (celle renvoyée par predict_proba), ou bien si le score de de décision est supérieur à 0 (decision_function)
#
#

# %%
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
X, y = make_classification(random_state=0)
y

# %%
classifier = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
#
# classifier.decision_function(X[:4]) // Not defined for DecisionTreeClassifier ?
# -> Claude : Les classifieurs à "partition dure" (comme les arbres) ne possède pas cette notion de "score de marge"
# qui représente "la confiance" de façon continue (ex: SVM, LDA, ...)

# %%
# Les proba d'appartenance à chaque classe, des 4 dernières observatons
classifier.predict_proba(X[:4])

# %%
# Les prédictions faites (par defaut : decision sur classe ayant proba > 0.5)
classifier.predict(X[:4])

# %% [markdown]
# Cette règle de décision  "P(y|X) > 0.5" peut sembler raisonable de prime abord mais elle peut s'avérer inéfficace dans certains, en particuler dans les cas de classe de désiquilibrées.
#
# Dans le scénario de prédiction de cancer sur des patients, on préférera modifier le seuil de manière à ne pas rater des cas positifs, même si cela rajoute des faux-posifif.

# %%
## Sur Example Gaussien du RCP-209 session1

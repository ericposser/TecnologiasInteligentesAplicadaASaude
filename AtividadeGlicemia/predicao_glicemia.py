# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("glicose_data.csv")

df = df.drop(columns=["Unnamed: 20", "Unnamed: 21"], errors="ignore")

df2 = df[["GLICEMIA", "SONO", "KCAL", "CARB"]]

df2 = df2.dropna()

df2

df2["KCAL"] = df2["KCAL"].str.strip().str.capitalize()
df2["CARB"] = df2["CARB"].str.strip().str.capitalize()
df2["GLICEMIA"] = df2["GLICEMIA"].str.strip().str.capitalize()

mapeamento_kcal = {"Abaixo": 0, "Recomendado": 1, "Acima": 2}
mapeamento_carb = {"Abaixo": 0, "Recomendado": 1, "Acima": 2}
df2["KCAL"] = df2["KCAL"].map(mapeamento_kcal)
df2["CARB"] = df2["CARB"].map(mapeamento_carb)

mapeamento_glicemia = {"Abaixo": 0, "Normal": 1, "Acima": 2}
df2["GLICEMIA"] = df2["GLICEMIA"].map(mapeamento_glicemia)

df2.head()

X = df2[["SONO", "KCAL", "CARB"]]
y = df2["GLICEMIA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# Acurácia geral
print("Acurácia:", accuracy_score(y_test, y_pred))

# Relatório detalhado
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=["Abaixo", "Normal", "Acima"]))

# Matriz de confusão
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Abaixo", "Normal", "Acima"],
            yticklabels=["Abaixo", "Normal", "Acima"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()
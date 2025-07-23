import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
# %% Leitura da base final
df = pd.read_csv('data/ABT_IBOVESPA.csv')
# %% Separação OUT OF TIME
oot = df.tail(30)
X_oot = oot.drop(columns=['Fechamento'])
y_oot = oot['Fechamento']
df = df.iloc[:-30]
# %% Separação treino/teste
X = df.drop(columns=['Fechamento'])
y = df['Fechamento']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# %% Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_oot_scaled = scaler.transform(X_oot)
# %% Regressão Logística
reg = LogisticRegression(class_weight='balanced', C=5)
reg.fit(X_train_scaled, y_train)
# %% Avaliação do modelo - TESTE
reg_predict = reg.predict(X_test_scaled)
reg_predict_proba = reg.predict_proba(X_test_scaled)
# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

# K-Fold com 6 divisões
kfold = KFold(n_splits=6, shuffle=True, random_state=42)

# Usar acurácia como métrica
accuracy = make_scorer(accuracy_score)

# Calcular acurácia com validação cruzada
result = cross_val_score(reg, X_train_scaled, y_train, cv=kfold, scoring=accuracy)

# Mostrar resultados
print(f'K-Fold Accuracy Scores: {result}')
print(f'Mean Accuracy for Cross-Validation K-Fold: {result.mean():.4f}')
# %%
df_predict = y_test.to_frame()
df_predict['Predict Reg Log'] = reg_predict
df_predict['Proba Reg Log'] = reg_predict_proba[:, 1]
# %%
# Matriz de Confusão
pd.crosstab(df_predict['Fechamento'], df_predict['Predict Reg Log'], normalize='index')
# %% Avaliação do modelo - OOT
reg_predict_oot = reg.predict(X_oot_scaled)
reg_predict_proba_oot = reg.predict_proba(X_oot_scaled)

df_predict_oot = y_oot.to_frame()
df_predict_oot['Predict Reg Log'] = reg_predict_oot
df_predict_oot['Proba Reg Log'] = reg_predict_proba_oot[:, 1]
# %%
# Matriz de Confusão
pd.crosstab(df_predict_oot['Fechamento'], df_predict_oot['Predict Reg Log'], normalize='index')
# %%
df_predict_oot
# %% Acurácias
print('Acurácia TESTE:', metrics.accuracy_score(y_test, reg_predict) * 100)
print('Acurácia TREINO:', metrics.accuracy_score(y_train, reg.predict(X_train_scaled)) * 100)
print('Acurácia OOT:', metrics.accuracy_score(y_oot, reg.predict(X_oot_scaled)) * 100)
# %% ROC AUC
roc_test = metrics.roc_curve(y_test, df_predict['Proba Reg Log'])
roc_train = metrics.roc_curve(y_train, reg.predict_proba(X_train_scaled)[:, 1])
roc_oot = metrics.roc_curve(y_oot, reg.predict_proba(X_oot_scaled)[:, 1])

auc_test = metrics.roc_auc_score(y_test, df_predict['Proba Reg Log'])
auc_train = metrics.roc_auc_score(y_train, reg.predict_proba(X_train_scaled)[:, 1])
auc_oot = metrics.roc_auc_score(y_oot, reg.predict_proba(X_oot_scaled)[:, 1])

plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title('Roc Curve')
plt.xlabel('1 - Especificidade')
plt.ylabel('Recall')
plt.legend([f'Teste: {auc_test:.2f}', f'Treino: {auc_train:.2f}', f'OOT: {auc_oot:.2f}'])
plt.show()
# %% Exporta modelo
joblib.dump(reg, 'modelo_log_reg_ibovespa.pkl')
joblib.dump(scaler, 'scaler_ibovespa.pkl')

# %% Visualização das previsões OOT
df_predict = y_oot.to_frame()
df_predict['Predict Reg Log'] = reg.predict(X_oot_scaled)
df_predict['Proba Reg Log'] = reg.predict_proba(X_oot_scaled)[:, 1]

import numpy as np
from scipy.special import expit

x = df_predict['Proba Reg Log']
y = df_predict['Predict Reg Log']
plt.scatter(x, y, label='Previsões')
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = expit((x_vals - 0.5) * 20)
plt.plot(x_vals, y_vals, color='red', label='Curva de probabilidade')
plt.grid(axis='x')
plt.title('Probabilidade vs Previsão')
plt.xlabel('Probabilidade prevista')
plt.ylabel('Classe prevista')
plt.legend()
plt.show()
# %%
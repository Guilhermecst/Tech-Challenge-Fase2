# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib
# %%
df = pd.read_csv('data/ABT_IBOVESPA.csv')
# %%
# OUT OF TIME
oot = df.tail(30)

X_oot = oot.drop(columns=['Fechamento', 'Data', 'Var%', 'Último'])
y_oot = oot['Fechamento']

# NOVO DATAFRAME SEM OOT
df = df.iloc[:-30]
# %%
X = df.drop(columns=['Fechamento', 'Data', 'Var%', 'Último'])
y = df['Fechamento']
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# %%
# %%
X_train.isna().sum()
# %%
print(f'Variáveis de treino: {len(X_train)}')
print(f'Variáveis resposta de treino: {len(y_train)}')

print(f'Variáveis de teste: {len(X_test)}')
print(f'Variáveis resposta de teste: {len(y_test)}')
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_oot_scaled = scaler.transform(X_oot)
# %%
# REGRESSÃO LOGÍSTICA
reg = LogisticRegression(class_weight='balanced')
reg.fit(X_train_scaled, y_train)
# %%
reg_predict = reg.predict(X_test_scaled)
reg_predict_proba = reg.predict_proba(X_test_scaled)
# %%
df_predict = y_test.to_frame()
df_predict['Predict Reg Log'] = reg_predict
df_predict['Proba Reg Log'] = reg_predict_proba[:,1]
df_predict
# %%
# MATRIZ DE CORRELAÇÃO
pd.crosstab(df_predict['Fechamento'], df_predict['Predict Reg Log'])
# %%
# ACURÁCIA TESTE
acc_test = metrics.accuracy_score(df_predict['Fechamento'], df_predict['Predict Reg Log'])
print(f'Acurácia do teste: {acc_test * 100:.2f}%')
# %%
# ACURÁCIA TREINO
acc_train = metrics.accuracy_score(y_train, reg.predict(X_train_scaled))
print(f'Acurácia do treino: {acc_train * 100:.2f}%')
# %%
# ACURÁCIA OOT
acc_oot = metrics.accuracy_score(y_oot, reg.predict(X_oot_scaled))
print(f'Acurácia do OOT: {acc_oot * 100:.2f}%')
# %%
roc_test = metrics.roc_curve(df_predict['Fechamento'], df_predict['Proba Reg Log'])
roc_train = metrics.roc_curve(y_train, reg.predict_proba(X_train_scaled)[:,1])
roc_oot = metrics.roc_curve(y_oot, reg.predict_proba(X_oot_scaled)[:,1])

auc_test = metrics.roc_auc_score(df_predict['Fechamento'], df_predict['Proba Reg Log'])
auc_train = metrics.roc_auc_score(y_train, reg.predict_proba(X_train_scaled)[:,1])
auc_oot = metrics.roc_auc_score(y_oot, reg.predict_proba(X_oot_scaled)[:,1])
# %%
import matplotlib.pyplot as plt

plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title('Roc Curve')
plt.xlabel('1 - Especificidade')
plt.ylabel('Recall')

plt.legend([f'Teste: {auc_test:.2f}', f'Treino: {auc_train:.2f}', f'OOT: {auc_oot:.2f}'])
plt.show()
# %%
joblib.dump(reg, 'modelo_log_reg_ibovespa.pkl')
# %%
df_predict = y_oot.to_frame()
df_predict['Predict Reg Log'] = reg.predict(X_oot_scaled)
df_predict['Proba Reg Log'] = reg.predict_proba(X_oot_scaled)[:,1]
# %%
import numpy as np
from scipy.special import expit  # Função sigmoide

x = df_predict['Proba Reg Log']
y = df_predict['Predict Reg Log']

plt.scatter(x, y, label='Previsões')

# Ordena os valores de x para a curva ficar suave
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = expit((x_vals - 0.5) * 20)

# Curva de probabilidade
plt.plot(x_vals, y_vals, color='red', label='Curva de probabilidade')

plt.grid(axis='x')
plt.title('Probabilidade vs Previsão')
plt.xlabel('Probabilidade prevista')
plt.ylabel('Classe prevista')
plt.legend()
plt.show()
# %%

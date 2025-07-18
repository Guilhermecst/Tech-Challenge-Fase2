# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# %%
df = pd.read_csv('data/ABT_IBOVESPA.csv')
# %%
# OUT OF TIME
oot = df.tail(30)

X_oot = oot.drop(columns=['Fechamento'])
y_oot = oot['Fechamento']

# NOVO DATAFRAME SEM OOT
df = df.iloc[:-30]
# %%
X = df.drop(columns=['Fechamento'])
y = df['Fechamento']
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
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
# ÁRVORE DE CLASSIFICAÇÃO
rf = RandomForestClassifier(class_weight='balanced', random_state=42, min_samples_split=10)
rf.fit(X_train_scaled, y_train)
# %%
rf_predict = rf.predict(X_test_scaled)
rf_predict_proba = rf.predict_proba(X_test_scaled)
# %%
df_predict = y_test.to_frame()
df_predict['Predict RF'] = rf_predict
df_predict['Proba RF'] = rf_predict_proba[:,1]
df_predict
# %%
# MATRIZ DE CORRELAÇÃO
pd.crosstab(df_predict['Fechamento'], df_predict['Predict RF'])
# %%
# ACURÁCIA TESTE
acc_test = metrics.accuracy_score(df_predict['Fechamento'], df_predict['Predict RF'])
print(f'Acurácia do teste: {acc_test * 100:.2f}%')
# %%
# ACURÁCIA TREINO
acc_train = metrics.accuracy_score(y_train, rf.predict(X_train_scaled))
print(f'Acurácia do treino: {acc_train * 100:.2f}%')
# %%
# ACURÁCIA OOT
acc_oot = metrics.accuracy_score(y_oot, rf.predict(X_oot_scaled))
print(f'Acurácia do OOT: {acc_oot * 100:.2f}%')
# %%
roc_test = metrics.roc_curve(df_predict['Fechamento'], df_predict['Proba RF'])
roc_train = metrics.roc_curve(y_train, rf.predict_proba(X_train_scaled)[:,1])
roc_oot = metrics.roc_curve(y_oot, rf.predict_proba(X_oot_scaled)[:,1])

auc_test = metrics.roc_auc_score(df_predict['Fechamento'], df_predict['Proba RF'])
auc_train = metrics.roc_auc_score(y_train, rf.predict_proba(X_train_scaled)[:,1])
auc_oot = metrics.roc_auc_score(y_oot, rf.predict_proba(X_oot_scaled)[:,1])
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

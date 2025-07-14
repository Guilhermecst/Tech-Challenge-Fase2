# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# %%
df = pd.read_csv('data/ABT_IBOVESPA.csv')
# %%
df_teste = df.tail(30)
df_treino = df.iloc[:-30]

# Separa X e y
X_train = df_treino.drop(columns=['Fechamento', 'Data', 'Var%', 'Último'])
y_train = df_treino['Fechamento']

X_test = df_teste.drop(columns=['Fechamento', 'Data', 'Var%', 'Último'])
y_test = df_teste['Fechamento']
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
# %%
# %%
# ÁRVORE DE CLASSIFICAÇÃO
dtc = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dtc.fit(X_train_scaled, y_train)
# %%
dtc_predict = dtc.predict(X_test_scaled)
dtc_predict_proba = dtc.predict_proba(X_test_scaled)
# %%
df_predict = y_test.to_frame()
df_predict['Predict DTC'] = dtc_predict
df_predict['Proba DTC'] = dtc_predict_proba[:,1]
df_predict
# %%
# MATRIZ DE CORRELAÇÃO
pd.crosstab(df_predict['Fechamento'], df_predict['Predict DTC'])
# %%
# ACURÁCIA
acc = metrics.accuracy_score(df_predict['Fechamento'], df_predict['Predict DTC'])
print(f'Acurácia do teste: {acc * 100:.2f}%')
# %%
acc_train = metrics.accuracy_score(y_train, dtc.predict(X_train_scaled))
print(f'Acurácia do treino: {acc_train * 100:.2f}%')
# %%
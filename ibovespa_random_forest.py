# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
# ACURÁCIA
acc = metrics.accuracy_score(df_predict['Fechamento'], df_predict['Predict RF'])
print(f'Acurácia do teste: {acc * 100:.2f}%')
# %%
acc_train = metrics.accuracy_score(y_train, rf.predict(X_train_scaled))
print(f'Acurácia do treino: {acc_train * 100:.2f}%')
# %%

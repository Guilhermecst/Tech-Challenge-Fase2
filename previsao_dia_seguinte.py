# %%
import pandas as pd
import joblib
# %%
# Carrega o modelo treinado
modelo = joblib.load('modelo_log_reg_ibovespa.pkl')

# Carrega o scaler
scaler = joblib.load('scaler_ibovespa.pkl')
# %%
# Carrega os dados
df = pd.read_csv('data/ABT_IBOV_AUTOMATICO.csv')
# %%
df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
# %%
# Colunas usadas no modelo
colunas_modelo = [
    'Abertura_Lag1', 'Máxima_Lag1', 'Mínima_Lag1',
    'Volume_Lag1', 'Último_Lag1', 'Fechamento_Lag1', 'Abertura_Media5', 'Máxima_Media5',
    'Mínima_Media5', 'Fechamento_Media5', 'Volume_Media5', 'Volatilidade5', 'Momentum_5'
    'Abertura_Media10', 'Máxima_Media10', 'Mínima_Media10',
    'Fechamento_Media10', 'Volume_Media10', 'Volatilidade10', 'Momentum_10','Abertura_Media15',
    'Máxima_Media15', 'Mínima_Media15', 'Fechamento_Media15',
    'Volume_Media15', 'Volatilidade15', 'Variação_Dia_Anterior_Lag1'
]
# %%
# Pega a última linha com as features calculadas
dados_hoje = df[colunas_modelo].iloc[[-1]]
# %%
# Aplica o scaler
dados_hoje_escalado = scaler.transform(dados_hoje)
# %%
# Faz a previsão
previsao = modelo.predict(dados_hoje_escalado)
# %%
if previsao[0] == 1:
    fechamento = 'Alta'
else:
    fechamento = 'Baixa'

print(f"Próximo fechamento: {fechamento}")
# %%
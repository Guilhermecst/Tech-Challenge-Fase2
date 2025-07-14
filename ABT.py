# %%
import pandas as pd
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('data/Dados Históricos - Ibovespa.csv', quotechar='"', sep=',')
# %%
df.head()
# %%
df.tail()
# %%
df.shape
# %%
df.info()
# %%
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
# %%
df['Vol.'] = df['Vol.'].str.replace(',', '.', regex=False)
# %%
def converter_volume(vol):
    if pd.isna(vol):
        return None  # trata valores nulos
    vol = vol.upper().strip()
    if vol.endswith('B'):
        return float(vol[:-1]) * 1_000_000_000
    elif vol.endswith('M'):
        return float(vol[:-1]) * 1_000_000
    elif vol.endswith('K'):
        return float(vol[:-1]) * 1_000
    else:
        try:
            return float(vol)  # caso não tenha letra no final
        except ValueError:
            return None  # erro de conversão

# Aplica no DataFrame
df['Vol.'] = df['Vol.'].apply(converter_volume)
# %%
df['Var%'] = df['Var%'].str.replace(',', '.', regex=False)
df['Var%'] = df['Var%'].str.replace('%', '', regex=False)
df['Var%'] = df['Var%'].astype(float)
# %%
df['Fechamento'] = df['Var%'].apply(lambda x: 1 if x > 0 else 0)
# 1 = Positivo
# 0 = Negativo
# %%
df['Fechamento'].value_counts()
# %%
# Separa os últimos 30 registros para teste
df = df.sort_values('Data')
# LAGS - 1 dia
df['Abertura_Lag1'] = df['Abertura'].shift(1)
df['Máxima_Lag1'] = df['Máxima'].shift(1)
df['Mínima_Lag1'] = df['Mínima'].shift(1)
df['Volume_Lag1'] = df['Vol.'].shift(1)
df['Fechamento_Lag1'] = df['Fechamento'].shift(1)

# LAGS - 3 dias
df['Abertura_Lag3_média'] = df['Abertura'].shift(1).rolling(window=3).mean()
df['Volume_Lag3_média'] = df['Vol.'].shift(1).rolling(window=3).mean()
df['Fechamento_Lag3_media'] = df['Fechamento'].shift(1).rolling(window=3).mean()

# Range e variação do dia anterior
df['Range_Lag1']  = (df['Máxima'] - df['Mínima']).shift(1)
df['Variação_Dia_Anterior_Lag1']  = (df['Abertura'] - df['Último']).shift(1)
# %%
df.isna().sum()
# %%
df = df.dropna().reset_index(drop=True)
# %%
df.head()
# %%
df.to_csv('data/ABT_IBOVESPA.csv', index=False)
# %%
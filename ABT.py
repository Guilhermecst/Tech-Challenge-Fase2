# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv('data/Dados Históricos - Ibovespa 2006.csv', quotechar='"', sep=',')
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
df.info()
# %%
df['Var%'] = df['Var%'].str.replace(',', '.', regex=False)
df['Var%'] = df['Var%'].str.replace('%', '', regex=False)
df['Var%'] = df['Var%'].astype(float)
# %%
df.info()
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

# Médias móveis de 5 dias
df['Abertura_Media5'] = df['Abertura'].rolling(window=5).mean()
df['Máxima_Media5'] = df['Máxima'].rolling(window=5).mean()
df['Mínima_Media5'] = df['Mínima'].rolling(window=5).mean()
df['Fechamento_Media5'] = df['Fechamento'].rolling(window=5).mean()
df['Volume_Media5'] = df['Vol.'].rolling(window=5).mean()

# Médias móveis de 10 dias
df['Abertura_Media10'] = df['Abertura'].rolling(window=10).mean()
df['Máxima_Media10'] = df['Máxima'].rolling(window=10).mean()
df['Mínima_Media10'] = df['Mínima'].rolling(window=10).mean()
df['Fechamento_Media10'] = df['Fechamento'].rolling(window=10).mean()
df['Volume_Media10'] = df['Vol.'].rolling(window=10).mean()

# Médias móveis de 15 dias
df['Abertura_Media15'] = df['Abertura'].rolling(window=15).mean()
df['Máxima_Media15'] = df['Máxima'].rolling(window=15).mean()
df['Mínima_Media15'] = df['Mínima'].rolling(window=15).mean()
df['Fechamento_Media15'] = df['Fechamento'].rolling(window=15).mean()
df['Volume_Media15'] = df['Vol.'].rolling(window=15).mean()

# Variação do dia anterior
df['Variação_Dia_Anterior_Lag1']  = (df['Abertura'] - df['Último']).shift(1)
# %%
df.isna().sum()
# %%
df = df.dropna().reset_index(drop=True)
# %%
df.head()
# %%
df.describe()
# %%
df['Fechamento'].value_counts()
# %%
sns.countplot(x='Fechamento', data=df, hue='Fechamento')
plt.grid(axis='y')
plt.title('Distribuição da variável resposta')
plt.legend(title='Target', bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.ylabel(None)
plt.show()
# %%
df.to_csv('data/ABT_IBOVESPA.csv', index=False)
# %%
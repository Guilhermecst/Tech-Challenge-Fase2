# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# %% Leitura do arquivo
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
        return None
    vol = vol.upper().strip()
    if vol.endswith('B'):
        return float(vol[:-1]) * 1_000_000_000
    elif vol.endswith('M'):
        return float(vol[:-1]) * 1_000_000
    elif vol.endswith('K'):
        return float(vol[:-1]) * 1_000
    else:
        try:
            return float(vol)
        except ValueError:
            return None

df['Vol.'] = df['Vol.'].apply(converter_volume)
# %%
df.info()
# %% Tratamento da coluna de variação
df['Var%'] = df['Var%'].str.replace(',', '.', regex=False)
df['Var%'] = df['Var%'].str.replace('%', '', regex=False)
df['Var%'] = df['Var%'].astype(float)
# %%
df.info()
# %%
df = df.sort_values('Data')
df['Fechamento'] = df['Var%'].apply(lambda x: 1 if x > 0.005 else 0).shift(-1)
# 1 = Positivo
# 0 = Negativo
# %%
df['Fechamento'].value_counts()
# %% LAGS - 1 dia
df['Abertura_Lag1'] = df['Abertura'].shift(1) # Abertura do dia anterior
df['Máxima_Lag1'] = df['Máxima'].shift(1) # Máxima do dia anterior
df['Mínima_Lag1'] = df['Mínima'].shift(1) # Mínima do dia anterior
df['Fechamento_Lag1'] = df['Fechamento'].shift(1) # Fechamento do dia anterior (alta/baixa)
df['Volume_Lag1'] = df['Vol.'].shift(1) # Volume do dia anterior
df['Último_Lag1'] = df['Último'].shift(1) # Preço do dia anterior

# Médias móveis de 5 dias
df['Abertura_Media5'] = df['Abertura'].rolling(window=5).mean() # Média da abertura dos últimos 5 dias de pregão
df['Máxima_Media5'] = df['Máxima'].rolling(window=5).mean() # Média da máxima dos últimos 5 dias de pregão
df['Mínima_Media5'] = df['Mínima'].rolling(window=5).mean() # Média da mínima dos últimos 5 dias de pregão
df['Fechamento_Media5'] = df['Fechamento'].rolling(window=5).mean() # # Média do fechamento dos últimos 5 dias de pregão
df['Volume_Media5'] = df['Vol.'].rolling(window=5).mean() # Média do volume dos últimos 5 dias de pregão
df['Último_Media5'] = df['Último'].rolling(window=5).mean() # Média do preço dos últimos 5 dias de pregão
df['Volatilidade5'] = df['Var%'].rolling(5).std() # Oscilação da variação percentual dos últimos 5 pregões

# Médias móveis de 10 dias
df['Abertura_Media10'] = df['Abertura'].rolling(window=10).mean() # Média da abertura dos últimos 10 dias de pregão
df['Máxima_Media10'] = df['Máxima'].rolling(window=10).mean()# Média da máxima dos últimos 10 dias de pregão
df['Mínima_Media10'] = df['Mínima'].rolling(window=10).mean() # Média da mínima dos últimos 10 dias de pregão
df['Fechamento_Media10'] = df['Fechamento'].rolling(window=10).mean() # Média do fechamento dos últimos 10 dias de pregão
df['Volume_Media10'] = df['Vol.'].rolling(window=10).mean() # Média do volume dos últimos 10 dias de pregão
df['Último_Media10'] = df['Último'].rolling(window=10).mean() # Média do preço dos últimos 10 dias de pregão
df['Volatilidade10'] = df['Var%'].rolling(10).std() # # Oscilação da variação percentual dos últimos 5 pregões

# Variação do dia anterior
df['Variação_Dia_Anterior_Lag1'] = (df['Abertura'] - df['Último']).shift(1) # Variação absoluta do dia anterior
df['Tendencia_5_10'] = df['Fechamento_Media5'] - df['Fechamento_Media10'] # Diferença entre médias móveis de 5 e 10 dias
# %%
df.drop(columns=['Máxima', 'Mínima', 'Data', 'Var%', 'Último', 'Abertura', 'Vol.'], inplace=True)
# %% Limpeza de NaNs e reset do índice
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
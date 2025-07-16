# %%
import yfinance as yf
import pandas as pd
# %%
# Define o ticker do IBOV e período
ticker = '^BVSP'
periodo = '30d'  # últimos 90 dias

# Baixa os dados
df = yf.download(ticker, period=periodo, interval='1d')
df.reset_index(inplace=True)

# Renomeia colunas para manter compatibilidade com seu código
df.rename(columns={
    'Date': 'Data',
    'Open': 'Abertura',
    'High': 'Máxima',
    'Low': 'Mínima',
    'Close': 'Último',
    'Volume': 'Vol.'
}, inplace=True)

# %%
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data')

# Cria variação percentual diária
df['Var%'] = df['Último'].pct_change() * 100

# Variável target
df['Fechamento'] = df['Var%'].apply(lambda x: 1 if x > 0 else 0)
# %%
# Lags
df['Abertura_Lag1'] = df['Abertura'].shift(1)
df['Máxima_Lag1'] = df['Máxima'].shift(1)
df['Mínima_Lag1'] = df['Mínima'].shift(1)
df['Volume_Lag1'] = df['Vol.'].shift(1)
df['Fechamento_Lag1'] = df['Fechamento'].shift(1)

# Médias móveis de 5 dias
for n in [5, 10, 15]:
    df[f'Abertura_Media{n}'] = df['Abertura'].rolling(window=n).mean()
    df[f'Máxima_Media{n}'] = df['Máxima'].rolling(window=n).mean()
    df[f'Mínima_Media{n}'] = df['Mínima'].rolling(window=n).mean()
    df[f'Fechamento_Media{n}'] = df['Fechamento'].rolling(window=n).mean()
    df[f'Volume_Media{n}'] = df['Vol.'].rolling(window=n).mean()

# Variação do dia anterior
df['Variação_Dia_Anterior_Lag1'] = (df['Abertura'] - df['Último']).shift(1)
# %%
# Remove colunas não necessárias
df_modelo = df.drop(columns=['Fechamento', 'Var%', 'Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.'])

# Remove nulos
df_modelo = df_modelo.dropna().reset_index(drop=True)

# Remove multiíndice nas colunas
df_modelo.columns = df_modelo.columns.get_level_values(-2)

# Remove índice nomeado (ex: 'Ticker') e reseta o índice numérico
df_modelo = df_modelo.reset_index(drop=True)
# %%
df_modelo.tail(1)
# %%
data_hoje = df_modelo.tail(1)
# Exporta para uso no modelo
data_hoje.to_csv('data/ABT_IBOV_AUTOMATICO.csv', index=False)
# %%

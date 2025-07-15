
# 📈 Previsão do Fechamento da Bolsa de Valores (B3 - IBOVESPA)

Este projeto tem como objetivo prever se o fechamento diário do índice **Bovespa (IBOVESPA)** será positivo ou negativo com base exclusivamente em seus próprios dados históricos. A ideia é construir um modelo de classificação binária que indique se o dia seguinte será de alta (1) ou de baixa (0), usando dados desde 2006 obtidos do site [Investing.com](https://br.investing.com/indices/bovespa-historical-data).

---

## 🗃️ Fonte dos Dados

Os dados foram coletados da seguinte URL:  
🔗 [https://br.investing.com/indices/bovespa-historical-data](https://br.investing.com/indices/bovespa-historical-data)

**Período:** Janeiro de 2006 até o presente.

### 📑 Dicionário de Dados

| Coluna       | Descrição                                                                 |
|--------------|---------------------------------------------------------------------------|
| `Data`       | Data da observação (formato dd.mm.yyyy)                                   |
| `Último`     | Valor de fechamento do índice no dia                                      |
| `Abertura`   | Valor de abertura do índice no dia                                        |
| `Máxima`     | Maior valor atingido pelo índice no dia                                   |
| `Mínima`     | Menor valor atingido pelo índice no dia                                   |
| `Vol.`       | Volume negociado no dia (com sufixos K, M ou B)                           |
| `Var%`       | Variação percentual do índice no dia em relação ao dia anterior           |

---

## 🔍 Pré-processamento e Engenharia de Atributos

O pré-processamento e criação da base analítica foram realizados no script `ABT.py`.

### ✔️ Etapas principais:

1. **Tratamento da coluna `Vol.`**  
   Conversão de strings com sufixos `K`, `M` e `B` para números reais:
   ```python
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
   ```

2. **Conversão da coluna `Var%`**  
   Transformada para `float`, removendo o símbolo `%`.

3. **Criação da variável target `Fechamento`**  
   - `1` se `Var% > 0` (alta)  
   - `0` se `Var% < 0` (baixa)

4. **Criação de variáveis defasadas (`lag`)**  
   Inclusão de variáveis do dia anterior para enriquecer o modelo:
   ```python
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
   ```

---

## 🤖 Modelagem Preditiva

### 🔬 Divisão dos dados

- **Out of Time (OOT):** últimos 30 dias da base.
- **Treino/Teste:** 80/20 do restante da base histórica.

### ⚙️ Modelos testados

- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression ✅ *[Melhor desempenho]*

### 🔎 Normalização

- Foi utilizado o `StandardScaler` para normalizar os dados antes do treinamento.

### 🏆 Melhor modelo

- **Regressão Logística** apresentou desempenho superior, com **acurácia consistente** entre treino e teste, sugerindo boa generalização.

---

## 🛠️ Tecnologias e Bibliotecas

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## 📂 Estrutura do Projeto

```
├── ABT.py                      # Script de tratamento e criação da base analítica
├── ibovespa_tree.py            # Treinamento e avaliação do modelo de árvore
├── ibovespa_random_forest.py   # Treinamento e avaliação do modelo de Random Forest
├── ibovespa_reg_log.py         # Treinamento e avaliação do modelo de Regressão Logística
├── data/                       # Dados brutos e tratados
├── README.md                   # Documentação do projeto
```

---

## 👤 Autor

Projeto desenvolvido por Guilherme Costa Silva.

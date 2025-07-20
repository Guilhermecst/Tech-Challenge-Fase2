
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
├── gera_dados_atuais.py        # Retorna um df com os dados do dia anterior
├── previsao_atual.py           # Preve o fechamento do dia atual
├── data/                       # Dados brutos e tratados
├── README.md                   # Documentação do projeto
```

---

## 📌 Para Testar

Siga o passo a passo abaixo para rodar o projeto localmente e prever se o IBOVESPA fechará em alta ou baixa no dia atual com base nos dados históricos:

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

> Substitua `seu-usuario/seu-repositorio` pela URL correta do seu repositório.

---

### 2. Crie e ative um ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

---

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> Caso ainda não tenha, crie um `requirements.txt` com as bibliotecas usadas no projeto, como:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

### 4. Gere os dados mais recentes

Execute o script que prepara os dados do dia anterior com as transformações adequadas:

```bash
python gera_dados_atuais.py
```

---

### 5. Execute a previsão

Rode o script de previsão com o modelo treinado:

```bash
python previsao_atual.py
```

A saída será algo como:

```
Previsão para o fechamento do dia: 1 (Alta)
```

ou

```
Previsão para o fechamento do dia: 0 (Baixa)
```

---

### ✅ Observações

- Verifique se os arquivos do modelo (`modelo_reg_log.pkl`) e do scaler (`scaler.pkl`) estão presentes no caminho correto.
- Caso deseje treinar novamente os modelos, utilize os scripts:
  - `ibovespa_tree.py`
  - `ibovespa_random_forest.py`
  - `ibovespa_reg_log.py`

---

## 👤 Autor

Projeto desenvolvido por Guilherme Costa Silva.

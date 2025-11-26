# Análise de PETR3 - Regressão Linear

## 1. Descrição dos dados

Período analisado: 2015-12-04 até 2016-09-26

Número de observações usadas: 212

Colunas utilizadas (Features):

Open, High, Low, Volume, Return, LogReturn, MA_5, MA_10, MA_21, MA_50, STD_5, STD_10, STD_21, STD_50, Volatility_21, HL_Range, OC_Range, DayOfWeek, Close_lag1, Close_lag2, Close_lag3, Return_lag1, Return_lag2, Return_lag3, Volume_lag1, Volume_lag2, Volume_lag3

## 2. Etapas realizadas

- Filtragem por ticker - Criação de variáveis derivadas (médias móveis, volatilidade, retornos, ranges) - Divisão temporal treino/teste (80/20) - Treinamento do modelo: Regressão Linear - Avaliação com MAE, RMSE e R2

## 3. Resultados do modelo

### Linear Regression

MAE: 0.367433  RMSE: 0.201207  R2: 0.718259  
![](predictions_linear_PETR3.png)

## 4. Conclusões

O modelo de regressão linear consegue capturar boa parte da variação diária do preço de PETR3, com R² = 0,72, indicando que explica ~72% da variabilidade.
O MAE ≈ 0,37 e RMSE ≈ 0,20 mostram que os erros médios são relativamente baixos em relação aos preços (cerca de 2–3% do valor), mas ainda existem pequenas diferenças entre previsões e valores reais.
Em resumo, o modelo é razoavelmente preciso para uma regressão linear simples, mas há espaço para melhorar com modelos mais complexos ou mais dados.
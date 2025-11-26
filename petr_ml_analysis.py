#!/usr/bin/env python3
"""
PETR_ML_analysis_LR_only.py (versão apenas com Regressão Linear)

Script para análise de séries temporais e modelo de ML (Regressão Linear)
Aplicável a um CSV com colunas: Date, Ticker, Open, High, Low, Close, Volume
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def read_and_prepare(csv_path, ticker):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'], dayfirst=True)
    except Exception as e:
        raise RuntimeError(f"Erro lendo CSV (parse_dates=['Date'], dayfirst=True): {e}")

    df.columns = [c.strip() for c in df.columns]

    if 'Ticker' not in df.columns:
        raise KeyError("Coluna 'Ticker' não encontrada no CSV.")

    df = df[df['Ticker'] == ticker].copy()
    if df.empty:
        raise ValueError(f"Nenhuma linha encontrada para o ticker '{ticker}' no arquivo {csv_path}.")

    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close']).diff()

    for w in [5, 10, 21, 50]:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'STD_{w}'] = df['Close'].rolling(window=w).std()

    df['Volatility_21'] = df['Return'].rolling(window=21).std()

    df['HL_Range'] = (df['High'] - df['Low']) / df['Open']
    df['OC_Range'] = (df['Open'] - df['Close']) / df['Open']

    df['DayOfWeek'] = df['Date'].dt.dayofweek

    for lag in [1, 2, 3]:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)
        df[f'Volume_lag{lag}'] = df['Volume'].shift(lag)

    df['Target'] = df['Close'].shift(-1)

    df = df.iloc[:-1].copy()

    df.dropna(inplace=True)

    if len(df) < 10:
        print(
            f"AVISO: depois de criar features e remover NaNs, restaram apenas {len(df)} linhas. "
            "Resultados podem ser instáveis. Considere fornecer mais dados."
        )

    return df


def build_feature_matrix(df, feature_cols):
    X = df[feature_cols].values
    y = df['Target'].values
    return X, y


def time_train_test_split(df, train_ratio=0.8):
    n = len(df)
    if n < 4:
        raise ValueError("Dataset muito pequeno para divisão treino/teste (precisa ter pelo menos 4 linhas).")
    train_end = int(n * train_ratio)
    if n - train_end < 2:
        train_end = max(2, n - 2)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()
    return train_df, test_df


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}, preds


def plot_predictions(dates, actual, preds, outpath, title='Actual vs Predicted'):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, preds, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(outpath))
    plt.close()


def generate_report(results, outdir, ticker, df, feature_cols):
    report_path = Path(outdir) / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Análise de {ticker} - Regressão Linear\n\n")
        f.write("## 1. Descrição dos dados\n\n")
        f.write(f"Período analisado: {df['Date'].min().date()} até {df['Date'].max().date()}\n\n")
        f.write(f"Número de observações usadas: {len(df)}\n\n")
        f.write("Colunas utilizadas (Features):\n\n")
        f.write(', '.join(feature_cols) + '\n\n')

        f.write("## 2. Etapas realizadas\n\n")
        f.write("- Filtragem por ticker - Criação de variáveis derivadas (médias móveis, volatilidade, retornos, ranges) - Divisão temporal treino/teste (80/20) - Treinamento do modelo: Regressão Linear - Avaliação com MAE, RMSE e R2\n\n")

        f.write("## 3. Resultados do modelo\n\n")
        
        model_name = 'Linear Regression'
        res = results[model_name]
        f.write(f"### {model_name}\n\n")
        f.write(f"MAE: {res['metrics']['mae']:.6f}  ")
        f.write(f"RMSE: {res['metrics']['rmse']:.6f}  ")
        f.write(f"R2: {res['metrics']['r2']:.6f}  \n")
        if 'plot' in res:
            f.write(f"![]({res['plot']})\n\n")

        f.write("## 4. Conclusões\n\n")
        f.write("(Preencha as conclusões quando revisar os resultados; por exemplo: desempenho do modelo, limitações e próximos passos.)\n")

    return report_path


def main():
    parser = argparse.ArgumentParser(description='Análise ML para PETR3/PETR4 (Apenas Regressão Linear)')
    parser.add_argument('--csv', required=True, help='Caminho para o arquivo CSV')
    parser.add_argument('--ticker', default='PETR3', help='Ticker a filtrar (PETR3 ou PETR4)')
    parser.add_argument('--outdir', default='results_lr_only', help='Diretório de saída')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proporção de treino (ex: 0.8)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        df = read_and_prepare(args.csv, args.ticker)
    except Exception as e:
        print("ERRO ao preparar dados:", e)
        sys.exit(1)

    feature_cols = [
        'Open', 'High', 'Low', 'Volume',
        'Return', 'LogReturn',
        'MA_5','MA_10','MA_21','MA_50',
        'STD_5','STD_10','STD_21','STD_50',
        'Volatility_21',
        'HL_Range','OC_Range',
        'DayOfWeek',
        'Close_lag1','Close_lag2','Close_lag3',
        'Return_lag1','Return_lag2','Return_lag3',
        'Volume_lag1','Volume_lag2','Volume_lag3'
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) == 0:
        print("ERRO: nenhuma feature disponível após filtragem. Colunas presentes:", list(df.columns))
        sys.exit(1)

    train_df, test_df = time_train_test_split(df, train_ratio=args.train_ratio)

    X_train, y_train = build_feature_matrix(train_df, feature_cols)
    X_test, y_test = build_feature_matrix(test_df, feature_cols)

    if X_train.shape[0] < 5:
        print("AVISO: poucas amostras de treino:", X_train.shape[0], "— resultados podem ser imprecisos.")

    results = {}

    # 1) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    metrics_lr, preds_lr = evaluate_model(lr, X_test, y_test)

    # save plot
    plot_path_lr = outdir / f'predictions_linear_{args.ticker}.png'
    plot_predictions(test_df['Date'], y_test, preds_lr, plot_path_lr, title='Linear Regression: Actual vs Predicted')

    results['Linear Regression'] = {'metrics': metrics_lr, 'plot': plot_path_lr.name}
    
    # save numeric results to CSV
    res_df = pd.DataFrame({
        'Date': test_df['Date'],
        'Actual': y_test,
        'Pred_LR': preds_lr,
    })
    res_df.to_csv(outdir / f'predictions_{args.ticker}.csv', index=False)

    # generate markdown report
    report_path = generate_report(results, outdir, args.ticker, df, feature_cols)

    print('\nResults saved to:', outdir.resolve())
    print('Report:', report_path)
    print('\nArquivos gerados:')
    for p in sorted(outdir.iterdir()):
        print('-', p.name)


if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from scipy import stats

# fpocket結果読み込み
df_fp = pd.read_csv('fpocket_gpcr_results.csv')

# CWxPモチーフ情報（correlation_final.csvから取得）
df_corr = pd.read_csv('correlation_final.csv')
print("correlation_final columns:", list(df_corr.columns))
print(df_corr.head(3).to_string())

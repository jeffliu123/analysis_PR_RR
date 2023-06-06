import numpy as np
import pandas as pd
MAE_sum = 0
RMSE_sum = 0
num_count = 0
path = r"/media/bestlab/Transcend/polar0330/jeffdark.CSV"
path1 = r"/home/bestlab/Desktop/final_com_v1/data_comp/Dark/jeff0330_Front/PR.csv"
df_estimate = pd.read_csv(path1, header=None)
df_estimate = df_estimate.iloc[1:,]
df_estimate = df_estimate.astype(int)
df_estimate = df_estimate.to_numpy()
print(df_estimate)
print(df_estimate[0][0])
df_polar = pd.read_csv(path, header=None)
# print(df_polar )
df_polar = df_polar.iloc[3:,2]
df_polar = df_polar.astype(int)
df_polar = df_polar.to_numpy()
print(df_polar)
print(df_polar[0])
for i in range(0,len(df_estimate)):
    MAE_sum += abs(df_estimate[i][0]-df_polar[i])
    RMSE_sum += pow(df_estimate[i][0]-df_polar[i],2)
    num_count += 1
print('MAE',round(MAE_sum/num_count,2))
print('RMSE',round(np.sqrt(RMSE_sum/num_count),2))
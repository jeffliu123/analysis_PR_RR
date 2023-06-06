import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import os
FS = 20
sliding_window = 5
round_precision = 1

def add_sliding(ans,peaks):
    ans.append(round((60 / ((peaks[1]-peaks[0]) / FS )),round_precision))
    ans.append(round((60 / ((peaks[2]-peaks[0]) / FS /2)),round_precision))
    ans.append(round((60 / ((peaks[3]-peaks[0]) / FS /3)),round_precision))
    return ans

def signal_process(RR_array):
    b, a = signal.butter(5, [0.1,0.8], 'bandpass', fs=FS) #bandpass filter
    filtedData = signal.filtfilt(b, a, RR_array) #bandpass filter
    peaks, _ = find_peaks(filtedData,height = 0.00001) #peak detection
    return peaks,filtedData

def data_augementation(s,count,peaks,ans,result):
    while s < len(peaks):
        if peaks[s] > count:
            result.append(ans[s])
            count += 20
            # print(s)
        else :
            s += 1
    return result

def cal_MAE_RMSE(result,df_estimate):
    MAE_sum = 0
    RMSE_sum = 0
    num_count = 0
    for i in range(10,len(result)):#len(result)
        MAE_sum += abs(df_estimate[0][i]-result[i])
        RMSE_sum += pow(df_estimate[0][i]-result[i],2)
        num_count += 1
    return MAE_sum,RMSE_sum,num_count

def main():
    RR_list = []
    x_list = []
    ans = []
    result = []
    count = 0
    s = 0
    path = r"/media/bestlab/Transcend/21th_c/Datasets/Dark/0105_jeff/Front/Vernier/0105_jeff_dark_0521_vernier_rb.csv"
    # path = r"/media/bestlab/Transcend/0330RR/jeff.csv"
    path_estimate = r"/home/bestlab/Desktop/final_com_v1/data_comp/Dark/jeff_Front/RR.csv"

    df_estimate = pd.read_csv(path_estimate, header=None)
    df_estimate = df_estimate.astype('float')
    df_estimate[0] = df_estimate[0].replace('NAN', 0)
    # print(df_estimate[0])
    # df_estimate = df_estimate.astype('int')
    print(df_estimate)
    print(len(df_estimate[0]))
    df = pd.read_csv(path, header=None)
    df = df.drop(df.columns[[0]], axis=1)
    df = df.drop(df.columns[[1]], axis=1)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 2))#normalize -1~1
    df = scaler.fit_transform(df)#normalize -1~1
    for i in range(0,len(df)):
        for j in range(0,1):
            RR_list.append(df[i][j])
    for k in range(0,len(df)):
        x_list.append(k)
    x_array = np.array(x_list)
    RR_array = np.array(RR_list)
    peaks,filtedData = signal_process(RR_array)
    # print(peaks)
    # print(len(peaks))
    ans = add_sliding(ans,peaks)
    for i in range(0,(len(peaks)-(sliding_window-1))):
        # ans.append(int(60 / ((peaks[i+(sliding_window-1)]-peaks[i]) / FS / (sliding_window-1))))
        ans.append(round(60 / ((peaks[i+(sliding_window-1)]-peaks[i]) / FS / (sliding_window-1)),round_precision))
    ans.insert(0,'NAN')
    # print(ans)
    # print(len(ans))
    result = data_augementation(s,count,peaks,ans,result)
    print(result)
    print(len(result))
    MAE_sum,RMSE_sum,num_count = cal_MAE_RMSE(result,df_estimate)
    print('MAE',round(MAE_sum/num_count,2))
    print('RMSE',round(np.sqrt(RMSE_sum/num_count),2))
    plt.title("Respiration Rate")
    plt.xlabel("frame")
    plt.ylabel("variation")
    plt.plot(x_array[peaks], filtedData[peaks], "x", color='red')
    plt.plot(filtedData, color='deepskyblue')
    plt.show()
main()

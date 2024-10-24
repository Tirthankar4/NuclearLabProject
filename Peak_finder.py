import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time

Map = {251.5: 81.5, 378.5: 121.782, 760.5: 244.697, 859.5: 276.5, 941.5: 303.5, 1069.5: 344.278, 1106.5: 356.5, 1192.5: 384.5,
       1277.5: 411.5, 1378.5: 444.5, 2420.5: 778.904, 2695.5: 867.380, 2996.5: 964.057, 3372.5: 1086.5, 3455.5: 1112.076,
       4374.5: 1408.013}

X_train = pd.DataFrame(Map.keys())
y_train = pd.Series(Map.values()).reindex(X_train.index)
x_train = list(Map.keys())

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#linear_slope = np.polyfit(x_train, y_train, 1)[0]

#energies = list(Map.values())

start_time = time.time()

def split_dataframe(df, n):
    
    df = df[df['Bin Center'] <= 5000]
    parts = []
    
    part_length = len(df) // n

    for i in range(n):
        start_index = i * part_length
        if i == n - 1:
            end_index = len(df)
        else:
            end_index = start_index + part_length
    
        part = pd.DataFrame(df.loc[start_index:end_index-1])
        parts.append(part)

    return parts 

def find_peaks(df_array, max_allowed_error = 0.01):

    energies = list(Map.values())
    peaks = []
    counts = []
    peaks_map = {}
    elements_used = len(energies)
    flag = False
    slope = linear_model.coef_[0]
    counter = 0
    avg = 0

    for i in range(len(df_array)):

        df = df_array[i]

        max_count = df['Bin Content'].max()

        df_1 = pd.DataFrame(df[df['Bin Content'] >= 0.2*max_count])

        df_1['Group'] = (df_1['Bin Center'].diff() > 1).cumsum()
        df_1 = df_1.loc[df_1.groupby('Group')['Bin Content'].idxmax()]
        df_1.drop(columns=['Group'], inplace = True)

        num_rows, _ = df_1.shape

        '''if num_rows > 5:
            
            df_1 = pd.DataFrame(df[df['Bin Content'] >= 0.6*max_count])

            df_1['Group'] = (df_1['Bin Center'].diff() > 1).cumsum()
            df_1 = df_1.loc[df_1.groupby('Group')['Bin Content'].idxmax()]
            df_1.drop(columns=['Group'], inplace = True)
            num_rows, _ = df_1.shape'''

        #print('rows = ', num_rows)
        #print(df_1)
        for j in range(num_rows):
            peaks.append(df_1.iloc[j]['Bin Center'])
            counts.append(df_1.iloc[j]['Bin Content'])

    for i in range(len(peaks)):
        a = peaks[i]
        energy = energies[0]
        diff = abs(energy/a - slope)
        for k in range(1, len(energies)):

            if abs((energies[k]/a - slope)) < diff:

                energy = energies[k]
                diff = abs(energy/a - slope)
        if slope - max_allowed_error < energy/a < slope + max_allowed_error:
            peaks_map[a] = energy

        keys_list = list(peaks_map.keys())
        #print(keys_list)

        if counter < 4:
            if i in range(len(keys_list)):
                avg += (1/4) * (energy / keys_list[i])
                #print(energy/list(peaks_map.keys())[i])
                counter += 1
        #print(avg)
        if i >= 4:
            #print('avg', avg)
            new_energy = energies[0]
            new_diff = abs(new_energy/a - avg)
            
            for l in range(1, len(energies)):

                if abs((energies[l]/a - avg)) < new_diff:

                    new_energy = energies[l]
                    new_diff = abs(new_energy/a - avg)
            #print(new_energy)
                
            if ((new_energy != energy) and (avg - 0.005 < new_energy/a < avg + 0.005)):
                peaks_map[a] = new_energy
                #print(new_energy)

    filtered_peaks_map = {}
    
    for key, value in peaks_map.items():
        count_index = list(peaks_map.keys()).index(key)
        count_value = counts[count_index]

        if value in filtered_peaks_map:
            if key > filtered_peaks_map[value]:
                filtered_peaks_map[value] = key
        else:
            filtered_peaks_map[value] = key

    final_peaks_map = {v: k for k, v in filtered_peaks_map.items()}

    return final_peaks_map

end_time = time.time()
time_taken = end_time - start_time

'''X = pd.read_csv(f"C:/Users/tirth/OneDrive/Desktop/Nuclear_project/CSV files/CL_01_E04.csv")
#print('Crystal 3')
parts = split_dataframe(X, 100)
peaks, peaks_map = find_peaks(parts, 0.01)
train = list(peaks_map.keys())
test = list(peaks_map.values())
print(peaks_map)

end_time = time.time()

for i in range(len(peaks_map)):
    print(f"{test[i] / train[i]:.5f}")

print('time taken is: ', end_time - start_time)'''

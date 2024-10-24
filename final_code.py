from Calibrator import *
from pathlib import Path
import subprocess

calibration_df = pd.DataFrame(columns = ['Intercept', 'Slope', 'Quadratic Coeffecient'])

folder_path = Path('C:/Users/tirth/OneDrive/Desktop/Nuclear_project/CSV files')

num_files = 0

for file_path in folder_path.iterdir():
    
    if file_path.is_file():
        
        X = pd.read_csv(file_path)
        slope, train, _, error = poly_fit(X, 2)
        print('length', len(train))
        print('error', error)

        calibration_df.loc[num_files] = [slope[0], slope[1], slope[2]]

        num_files += 1
        
#print(calibration_df)

import ROOT
import numpy as np
import pandas as pd
#import csv

file = ROOT.TFile("C:/Users/tirth/OneDrive/Desktop/Nuclear_project/EuBa_raw.root", 'READ')

H1 = file.Get("CL_03_E04")
data = []
nBins = H1.GetNbinsX()
for i in range(1, nBins):
    binCenter = H1.GetBinCenter(i);
    binContent = H1.GetBinContent(i);
    array = [binCenter, binContent]
    data.append(array)

X1 = pd.DataFrame(data, columns = ['Bin Center' , 'Bin Content'])
filename = 'C:/Users/tirth/OneDrive/Desktop/Nuclear_project/CSV files/CL_03_E04.csv'
X1.to_csv(filename, index = False)

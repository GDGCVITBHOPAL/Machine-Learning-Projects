import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
df = pd.read_csv('star.csv', parse_dates=True)  
    
df1 = pd.read_csv('galaxy.csv', parse_dates=True) 
    
plot = plt.figure().gca(projection='3d')
plot.scatter(df['stdDevYAbsGrad'], df['MeanGradX'], df['MeanGradY'], color = '#0000FF')
plot.set_xlabel('stdDevYAbsGrad')
plot.set_ylabel('Xmeangrad')
plot.set_zlabel('Ymeangrad')
    
    
plot.scatter(df1['stdDevYAbsGrad'], df1['MeanGradX'], df1['MeanGradY'], color = '#FF0000')
plot.set_xlabel('stdDevYAbsGrad')
plot.set_ylabel('Xmeangrad')
plot.set_zlabel('Ymeangrad')
    
plt.show()

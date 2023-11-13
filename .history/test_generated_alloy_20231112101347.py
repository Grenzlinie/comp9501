import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

r_data = pd.read_excel('Data_Warehouse/data.xlsx').iloc[:699, 1:7].values
g_data = pd.read_excel('Results/generate_alloys.xlsx').values

def reduce_dimension_and_plot(r_data, g_data):   
    # Perform dimensionality reduction to 2 dimensions
    pca = PCA(n_components=2)
    pca.fit(r_data)
    reduced_data1 = pca.transform(r_data)
    reduced_data2 = pca.transform(g_data)
    
    # Plot the scatter plot of the reduced data
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1])
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(['Real Data', 'Generated Data'])
    plt.title('Scatter Plot of Reduced Data')
    plt.show()

def analyse_the_data_distrubtion(r_data, g_data):
    plt.imshow(r_data.T, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.yticks([0, 1, 2, 3, 4, 5], ['Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu'])
    plt.xlabel('Compositions')
    plt.ylabel('Elements')
    plt.title('Heatmap of Real Data')
    plt.colorbar()  # 添加heatmap值的轴
    plt.show()
    plt.imshow(g_data.T, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.yticks([0, 1, 2, 3, 4, 5], ['Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu'])
    plt.xlabel('Compositions')
    plt.ylabel('Elements')
    plt.title('Heatmap of Generated Data')
    plt.colorbar()  # 添加heatmap值的轴
    plt.show()
analyse_the_data_distrubtion(r_data,g_data)




# reduce_dimension_and_plot(r_data, g_data)



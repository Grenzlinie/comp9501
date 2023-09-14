import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
data = data.iloc[0:699]
scaler = MinMaxScaler()
data.iloc[:, 7:] = scaler.fit_transform(data.iloc[:, 7:])
cat_data = np.concatenate((data.iloc[:, 8].values.reshape(-1, 1), data.iloc[:, 11].values.reshape(-1, 1)), axis=1)
tensor_data = torch.tensor([[100, 0], [0, 100], [0.1, 0.7]], dtype=torch.float32)


def sample_alloy_compositions(tensor_data):
    generator = torch.load('Results/generator_n.pth', map_location=torch.device('cpu'))
    generator.eval()
    with torch.no_grad():
        output = generator(tensor_data, torch.FloatTensor(np.random.normal(0, 1, (3, 100))))
        output[output < 0.001] = 0
        output = output / torch.sum(output, dim=1, keepdim=True)
        # row_sums = torch.sum(output, dim=1)
        # print("row_sums:", row_sums)
        print("output:", output)
        # Store the output to a CSV file
        # df = pd.DataFrame(output.numpy())
        # df.to_excel('Results/generate_alloys.xlsx', index=False, header='Fe,Ni,Co,Cr,V,Cu'.split(','))

def draw_correlation_map(cat_data):
    x = cat_data[:, 0]
    y = cat_data[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title("Correlation between x and y")
    ax.set_xlabel("Atomic Radius")
    ax.set_ylabel("Density")
    plt.savefig('Results/Relation_between_AtomRadius_and_Density.jpg')
    plt.show()


def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    
    
def calculate_distances():
    # 两个五维向量
    a = pd.read_excel('Results/generate_alloys.xlsx').values
    b = data.iloc[:, 1:7].values
    # 计算欧氏距离
    euclidean_distance = []
    for i in range(699):
        euclidean_distance.append(np.linalg.norm(a[i] - b[i]))
    print("平均欧氏距离:", np.mean(euclidean_distance))
    import torch.nn.functional as F

    def kl_divergence(x, y):
        kl_div = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='mean')
        print("KL散度:", kl_div.item())
    kl_divergence(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))


    

sample_alloy_compositions(tensor_data)
# calculate_distances()




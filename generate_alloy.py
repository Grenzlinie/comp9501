import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

def sample_alloy_compositions(tensor_data):
    generator = torch.load('Results/generator.pth', map_location=torch.device('cpu'))
    generator.eval()
    with torch.no_grad():
        output = generator(tensor_data, torch.FloatTensor(np.random.normal(0, 1, (699, 50))))
        output[output < 0.001] = 0
        output = output / torch.sum(output, dim=1, keepdim=True)

        # row_sums = torch.sum(output, dim=1)
        # print("row_sums:", row_sums)
        # print("output:", output)

        # Store the output to a excel file
        df = pd.DataFrame(output.numpy())
        df.to_excel('Results/generate_alloys.xlsx', index=False, header='Fe,Ni,Co,Cr,V,Cu'.split(','))


# def calculate_distances():
#     # Two 5 dimension vectors
#     a = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl').values
#     b = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl').iloc[0:699, 7:].values
#     # Calculate Euclidean distance
#     euclidean_distance = []
#     for i in range(699):
#         euclidean_distance.append(np.linalg.norm(a[i] - b[i]))
#     print("Euclidean distance:", np.mean(euclidean_distance))


#     def kl_divergence(x, y):
#         kl_div = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='mean')
#         print("KL divergence:", kl_div.item())
#     kl_divergence(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))



data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
data = np.array(data.iloc[0:699, 7:])
tensor_data = torch.tensor(data, dtype=torch.float32)
sample_alloy_compositions(tensor_data)
# calculate_distances()




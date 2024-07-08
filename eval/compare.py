import torch

# Load the files
file1_path = '/datadrive/yingwei/LangSplat_prev_generate_maps/eval/0_valid_map_level_1.pt'
file2_path = '/datadrive/yingwei/LangSplat_prev_generate_maps/eval/0_valid_map_level_2.pt'

data1 = torch.load(file1_path)
data2 = torch.load(file2_path)

# Compare the two files
are_equal = torch.equal(data1, data2)

print(f"Are the two files the same? {are_equal}")

#!/bin/bash

# Step 1: Update evaluate_iou_loc.py for the initial case
sed -i "108s|return .*|return torch.save(valid_map, f'{scene_index}_valid_map_level_{lvl}.pt')|" /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/evaluate_iou_loc_vq.py

# Step 2: Update openclip_encoder.py for the initial case
sed -i "161|return .*|return positive_vals|" /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/openclip_encoder.py

# Step 3: Execute eval.sh
sh eval.sh

# Loop for 4 iterations
for i in 0 1 2 3
do
    # Update evaluate_iou_loc.py with the loop variable
    sed -i "108s|return .*|return torch.save(valid_map, f'{scene_index}_valid_map_level_{lvl}_neg_${i}.pt')|" /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/evaluate_iou_loc_vq.py
    
    # Update openclip_encoder.py with the loop variable
    sed -i "161|return .*|return negative_vals[...,$i].unsqueeze(1)|" /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/openclip_encoder.py
    
    # Execute eval.sh
    sh eval.sh
done

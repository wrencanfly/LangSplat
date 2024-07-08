#!/bin/bash



# dim discover WORKFLOW

# !!! CLEARN THE PREV OUTPUT BEFORE START NEW DIM !!!
# EDIT FILES:
# >> rasterization edit <<
# /datadrive/yingwei/LangSplat/submodules/langsplat-rasterization
# Step 1: delete build file and egg info
# Step 2: cuda_rasterizer/config.h  change NUM_CHANNELS_language_feature
# Step 3: pip uninstall diff-gaussian-rasterization -y & pip install submodules/langsplat-rasterization/
#
# >> eval edit <<
# Step 4: go to eval/eval.sh change CASE_AC_DIM 




# Global variable
language_feature_dim=3
#case_name="ramen"

# # # Define the file name for saving runtime data
# runtime_file="runtime_${language_feature_dim}.txt"

# # Create or clear the runtime file
# echo "Runtime Information for Feature Dimension ${language_feature_dim}" > "$runtime_file"
# echo "-------------------------------------------" >> "$runtime_file"

# echo "Training autoencoder:" >> "$runtime_file"
# start_time=$(date +%s)
# # Train auto-encoder
# cd autoencoder
# # python train.py --dataset_path ../dataset/lerf_ovs/${case_name} --encoder_dims 256 128 64 32 ${language_feature_dim} --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ${case_name}
# python train.py --dataset_path ../dataset/lerf_ovs/${case_name} --encoder_dims 256 ${language_feature_dim} --decoder_dims 256 256 512 --lr 0.0007 --dataset_name ${case_name}

# # end_time=$(date +%s)
# # duration=$((end_time - start_time))

# # echo "Training for autoencoder took $duration seconds." >> "$runtime_file"

# # # Get the n-dims language feature of the scene
# python test.py --dataset_path ../dataset/lerf_ovs/${case_name} --dataset_name $case_name --encoder_dims 256 ${language_feature_dim} --decoder_dims 256 256 512


# # cd ..

# python train.py -s dataset/lerf_ovs/teatime -m dataset/lerf_ovs/teatime/output/teatime --start_checkpoint dataset/lerf_ovs/teatime/output/teatime/chkpnt30000.pth --feature_level 1 --language_feature_dim $language_feature_dim
# python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/teatime/output/teatime_1 --include_feature --language_feature_dim $language_feature_dim
## Start to train the Langsplat
# echo "Training Langsplat:" >> "$runtime_file"

# start_time=$(date +%s)
# for level in 1 2 3
# do
#     #echo "- Running Level $level -" >> "$runtime_file"
#     python train.py -s dataset/lerf_ovs/teatime -m dataset/lerf_ovs/teatime/output/teatime --start_checkpoint dataset/lerf_ovs/teatime/output/teatime/chkpnt30000.pth --feature_level ${level}
#     #echo "- Level $level done -" >> "$runtime_file"
# done


# end_time=$(date +%s)
# duration=$((end_time - start_time))

# echo "Training for Langsplat took $duration seconds." >> "$runtime_file"


# Render
# echo "Rendering:" >> "$runtime_file"
# start_time=$(date +%s)
case_name="figurines"
for level in 1
do
    # Render RGB (commented out per script)
    # python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/teatime/output/teatime_${level} --language_feature_dim $language_feature_dim

    # Render language features
    python render.py -m /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/dataset/lerf_ovs/${case_name}/output/${case_name}_${level} --include_feature 

    # python _mytest.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/waldo_kitchen/output/waldo_kitchen_3 --include_feature 
    # python _mytest.py -m dataset/lerf_ovs/figurines/output/figurines_1 --include_feature 
done
# end_time=$(date +%s)
# duration=$((end_time - start_time))

# echo "Rending for Langsplat took $duration seconds." >> "$runtime_file"

# # Eval
# cd eval
# sh eval.sh


# python _mytest.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/figurines/output/figurines_3 --include_feature 
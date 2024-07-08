#!/bin/zsh
#conda activate langsplat
# pip uninstall diff-gaussian-rasterization -y
# pip install submodules/langsplat-rasterization/
# Define the range of n values
for n in  $(seq 3 4 512)
do
    echo "Processing for n = $n"

    # Step 1: Update __init__.py
    sed -i "91s/n = [0-9]*/n = $n/" /datadrive/yingwei/LangSplat/gaussian_renderer/__init__.py

    # Step 2: Update config.h
    sed -i "16s/#define NUM_CHANNELS_language_feature [0-9]*/#define NUM_CHANNELS_language_feature $n/" /datadrive/yingwei/LangSplat/submodules/langsplat-rasterization/cuda_rasterizer/config.h

    # Step 3: Uninstall and install diff-gaussian-rasterization
    pip uninstall diff-gaussian-rasterization -y
    pip install submodules/langsplat-rasterization/

    # Step 4: Run the process.sh script and capture its output
    output=$(sh /datadrive/yingwei/LangSplat/process.sh 2>&1 | grep "Time taken to render:")
    
    # Step 5: Record the output to the file
    if [ ! -z "$output" ]; then
        echo "n = $n : $output" >> /datadrive/yingwei/LangSplat/render_time.info
    else
        echo "n = $n : No render time recorded" >> /datadrive/yingwei/LangSplat/render_time.info
    fi
done

echo "Batch processing complete."

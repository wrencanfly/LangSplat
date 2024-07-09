# [CVPR2024 Highlight] LangSplat: 3D Language Gaussian Splatting 

- [x] release more preprocessed dataset and the pretrained model (coming soon)
- [x] release the code of the eval

This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.

branch explanation:
- **prev_generate_maps**: generate 3d query sim maps and render them
  
  > The main changes happened in the **eval folder** and **gaussian_renderer folder**. And the **render.py** file
- **eval_only**: only eval the 3d query's output, not 2d
- **langplat_vq**: generate vq codebook and indices

![image](https://github.com/wrencanfly/LangSplat/assets/56505931/d3d1841c-1ffb-457f-ae3f-0fde15bb9242)
![image](https://github.com/wrencanfly/LangSplat/assets/56505931/ee12899c-2f73-4d6b-a496-669819cd76e0)

**Attention**
This repo has frequent hard codes and duplicate codes. For high-level learning, understanding, and semi-test only. DON'T TRY TO RUN IT!!!!

Usage:(Try not to run the code.)
without vq:
- **prev_generate_maps**:
    1. use **_mytest.py** to generate language_feats_dim3_tensor_1.pt, and copy them to the **eval/ folder**.
    2. go to **eval** folder, **sh eval.sh** to generate 0_valid_map_level_1_neg_0.pt, etc. This is the sim map, show be 1 pos and 4 neg
    3. render:
       
       3.1 go back to base folder, edit the **render.py** file, hard code exist here. Modify the corresponding parts at **meta_data.info**. Then **sh process.sh**
       
       3.2 hardcode exist here - **LangSplat/gaussian_renderer/__init__.py**. modify it
    5. output should be **LangSplat/dataset/lerf_ovs/figurines/output/figurines_1/train**.

- **eval_only**
  1. only the eval folder changed. Hardcode here.

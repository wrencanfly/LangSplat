CASE_NAME="teatime"
LANG_DIM=3
# path to lerf_ovs/label
gt_folder="../dataset/lerf_ovs/label"

root_path="../"

python evaluate_iou_loc.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/dataset/lerf_ovs/teatime/output \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 ${LANG_DIM} \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder}
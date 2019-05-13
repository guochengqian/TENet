python -u test_folder.py \
--pretrained_model /data/sony/output/model_log/p1trinity/ablation_study/demo-df2kx6-6-3-64-2-rrdb_checkpoint_1096.0k.path \
--model demo --block_type rrdb --n_resblock 6 --channels 64 --datatype uint8 --crop_scale 1 --img_type raw \
--test_path /data/datasets/RawSR20181001/sim_test/noisy_input/sigma20/kodak \
--save_path /data/datasets/RawSR20181001/sim_test/ablation_study_temp/temp1


python -u test_folder.py  --pretrained_model /data/sony/output/model_log/p1trinity/ablation_study/denorgb-dn-df2kx6-6-3-64-2-rrdb_checkpoint_830.0k.path --model denorgb --block_type rrdb --n_resblock 6 --channels 64  --datatype uint8 --crop_scale 1 --img_type raw --denoise --sigma 10 --test_path /data/datasets/RawSR20181001/sim_test/ablation_study_temp/temp1 --save_path /data/datasets/RawSR20181001/sim_test/ablation_study_temp/temp2

python -u test_folder.py  --pretrained_model /data/sony/output/model_log/p1trinity/ablation_study/srrgb-df2kx6-6-3-64-2-rrdb_checkpoint_1100.0k.path --model srrgb --block_type rrdb --n_resblock 6 --channels 64  --datatype uint8 --crop_scale 1 --img_type raw --scale 2 --test_path /data/datasets/RawSR20181001/sim_test/ablation_study_temp/temp2 --save_path /data/datasets/RawSR20181001/sim_test/ablation_study_temp/temp3
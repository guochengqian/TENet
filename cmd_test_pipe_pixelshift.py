"""
The testing file for rethinking the pipeline
Instruction:
# in_type:  current test in_type
# out_type:  current test out_type
# model:  current model
# pre_in_type:  previous test in_type
# pre_out_type:  previous test out_type
# pre_model: previous model
# intermidiate intermidiate state default false (false to read from dnd , true to read from previous)
Created by Yuanhao Wang
"""
import os
import argparse
import os.path as osp

PIPE_MAP = {
    'dn-dm-sr': 0,
    'dn-sr-dm': 1,
    'dm-dn-sr': 2,
    'dm-sr-dn': 3,
    'sr-dm-dn': 4,
    'sr-dn-dm': 5
}

parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
parser.add_argument('--pipeline', type=str, required=True,
                    help='which pipeline to evaluate'
                         'dn-dm-sr: 0'
                         'dn-sr-dm: 1'
                         'dm-dn-sr: 2'
                         'dm-sr-dn: 3'
                         'sr-dm-dn: 4'
                         'sr-dn-dm: 5'
                         '')
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--pretrain_dataset', type=str, required=True)
parser.add_argument('--test_dataset', type=str, default='pixelshit')
parser.add_argument('--noise_model', type=str, default='gp')
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--block', type=str, default='rrdb')
parser.add_argument('--n_blocks', type=int, default=6)
args = parser.parse_args()

pipeline = args.pipeline
args.save_dir = osp.join(args.pretrain,f'results_{args.test_dataset}_{args.noise_model}x{args.scale}', args.pipeline)

n_steps = 3
if pipeline == 'dn-dm-sr':
    # cmd1 denoise
    # cmd2 demosaic
    # cmd3 super resolution
    # DN-> DM->SR
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd3 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'dn-sr-dm':
    # cmd1 denoise
    # cmd2 super resolution
    # cmd3 demosaic
    # DN-> SR->DM
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_raw --out_type raw --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd3 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'raw', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'dm-dn-sr':
    # cmd1 demosaic
    # cmd2 denoise
    # cmd3 super resolution
    # DM -> DN -> SR
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd3 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_linrgb --out_type linrgb --pre_in_type noisy_linrgb --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'dm-sr-dn':
    # cmd1 demosaic
    # cmd2 super resolution
    # cmd3 denoise
    # DM -> SR -> DN
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd3 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_linrgb --out_type linrgb --pre_in_type lr_linrgb --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'sr-dm-dn':
    # cmd1 super resolution
    # cmd2 demosaic
    # cmd3 denoise
    # SR -> DM -> DN
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd3 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'noisy_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'sr-dn-dm':
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_raw --out_type raw --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd3 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'raw', 'None', 'linrgb']))
    mapping_path = osp.join('result')


# Partial Joint method
elif pipeline == 'dn-dm+sr' or pipeline == 'dn-sr+dm':
    # cmd1 super resolution
    # cmd2 denoise
    # SR -> DN -> DM
    n_steps = 2
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model tenet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {}  --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks*2, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'lr_raw', 'None', 'linrgb']))
    mapping_path = osp.join('result')


elif pipeline == 'dn+dm-sr':
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    n_steps = 2
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_raw --out_type linrgb --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type lr_linrgb --out_type linrgb --pre_in_type noisy_raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {}  --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'dn+sr-dm':
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    n_steps = 2
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    cmd2 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type raw --out_type linrgb --pre_in_type noisy_lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'raw', 'None', 'linrgb']))
    mapping_path = osp.join('result')

elif pipeline == 'e2e-dn+sr+dm':
    # fully end2end method
    # todo: revise here.
    n_steps = 1
    
    cmd1 = "python test_pipe_pixelshift.py --phase test --batch_per_gpu 1 --in_type noisy_lr_raw --out_type lr_raw --pre_out_type raw --model resnet --pretrain {} --dataset {} --test_dataset {} --scale {} --block {} --n_blocks {} --noise_model {} --save_dir {}".format(
        args.pretrain, args.pretrain_dataset, args.test_dataset, args.scale, args.block, args.n_blocks, args.noise_model, args.save_dir)
    final_save_path = osp.join('-'.join(['result', 'lr_raw', 'None', 'linrgb']))
    mapping_path = osp.join('result')


else:
    raise NotImplementedError('please check your pipeline args, it is not supported')

cmd_final = "ln -s {} {}".format(osp.join(args.save_dir, final_save_path), osp.join(args.save_dir, mapping_path))

print(f'\ncmd1: {cmd1} \n cmd2: {cmd2} \n')
if n_steps > 2:
    print(f'\ncmd3: {cmd3} \n')
print(f'\ncmd_final: {cmd_final} \n')

print(f'\n\n\n========= step 1: {cmd1} \n')
return_value = os.system(cmd1)

print(f'\n\n\n========= step 2: {cmd2} \n')
return_value = os.system(cmd2)

if n_steps > 2:
    print(f'\n\n\n========= step 3: {cmd3} \n')
    return_value = os.system(cmd3)

print(f'\n\n\n========= step 4: {cmd_final} \n')
os.system(f'rm -rf {mapping_path}')
os.system(cmd_final)

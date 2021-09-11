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

"""
import os
import argparse
import os.path as osp

cmd = "git --version"
return_value = os.system(cmd)
print('returned value :', return_value)

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
parser.add_argument('--pretrain_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, default=None,
                    help='if None, save to the result director under pretrain_dir')
parser.add_argument('--test_data', type=str,
                    default='data/benchmark/pixelshift200/pixelshift200_noisy_lr_raw_rgb_x2.pt',
                    help='path to the input data')
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--block_type', type=str, default='res')
parser.add_argument('--n_blocks', type=int, default=18)

args = parser.parse_args()

pipeline = args.pipeline
if args.save_dir is None:
    test_dataset = osp.basename(args.test_data).split('_')[0]
    args.save_dir = osp.join(args.pretrain_dir, f'result_{test_dataset}')

n_steps = 3

if pipeline == 'dn-dm-sr':
    # cmd1 denoise
    # cmd2 demosaic
    # cmd3 super resolution
    # DN-> DM->SR

    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd3 = "python test_pipe_pixelshift.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'dn-sr-dm':
    # cmd1 denoise
    # cmd2 super resolution
    # cmd3 demosaic
    # DN-> SR->DM
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type lr_raw --out_type raw --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet  --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    cmd3 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'raw', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'dm-dn-sr':
    # cmd1 demosaic
    # cmd2 denoise
    # cmd3 super resolution
    # DM -> DN -> SR
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type noisy_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd3 = "python test_pipe_pixelshift.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type noisy_linrgb --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'dm-sr-dn':
    # cmd1 demosaic
    # cmd2 super resolution
    # cmd3 denoise
    # DM -> SR -> DN
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    cmd3 = "python test_pipe_pixelshift.py --phase test --in_type noisy_linrgb --out_type linrgb --pre_in_type lr_linrgb --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'sr-dm-dn':
    # cmd1 super resolution
    # cmd2 demosaic
    # cmd3 denoise
    # SR -> DM -> DN
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd3 = "python test_pipe_pixelshift.py --phase test --in_type noisy_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'noisy_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'sr-dn-dm':
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type noisy_raw --out_type raw --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd3 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet  --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'raw', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')


# Partial Joint method
elif pipeline == 'dn-dm+sr' or pipeline == 'dn-sr+dm':
    # cmd1 super resolution
    # cmd2 denoise
    # SR -> DN -> DM
    n_steps = 2
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type lr_raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model tenet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {}  --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_raw', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')


elif pipeline == 'dn+dm-sr':
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    n_steps = 2
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_raw --out_type linrgb --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type noisy_raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {}  --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_linrgb', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'dn+sr-dm':
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    n_steps = 2
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type raw --out_type linrgb --pre_in_type noisy_lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'raw', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')

elif pipeline == 'dn-sr+dm':
    n_steps = 2
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_lr_raw --out_type lr_raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    cmd2 = "python test_pipe_pixelshift.py --phase test --in_type lr_raw --out_type linrgb --pre_in_type noisy_lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain_dir {} --save_dir {} --test_data {}  --scale {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.scale, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_raw', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')


elif pipeline == 'e2e-dn+sr+dm':
    # fully end2end method
    n_steps = 1
    args.save_dir = osp.join(args.save_dir, f'{pipeline}-SR{args.scale}')
    cmd1 = "python test_pipe_pixelshift.py --phase test --in_type noisy_lr_raw --out_type lr_raw --pre_out_type raw --model resnet --pretrain_dir {} --save_dir {} --test_data {} --block_type {} --n_blocks {}".format(
        args.pretrain_dir, args.save_dir, args.test_data, args.block_type, args.n_blocks)
    final_save_path = osp.join(args.save_dir, '-'.join(['result', 'lr_raw', 'None', 'linrgb']))
    mapping_path = osp.join(args.save_dir, 'result')


else:
    raise NotImplementedError('please check your pipeline args, it is not supported')

cmd_final = "ln -s {} {}".format(final_save_path, mapping_path)

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

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
cmd = "git --version"
return_value = os.system(cmd)
print('returned value :', return_value)

parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
parser.add_argument('--pipeline', type=int, required=True,
                    help='which pipeline to evaluate'
                         'DN-> DM->SR: 0'
                         'DN-> SR->DM: 1'
                         'DM -> DN -> SR: 2'
                         'DM -> SR -> DN: 3'
                         'SR -> DM -> DN: 4'
                         'SR -> DN -> DM: 5'
                         '')
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--scale', type=int, default=2)

args = parser.parse_args()
pipeline = args.pipeline
if args.save_dir is None:
    args.save_dir = os.path.join(args.pretrain, 'pipe_result')

if pipeline == 0:
    # cmd1 denoise
    # cmd2 demosaic
    # cmd3 super resolution
    # DN-> DM->SR
    cmd1 = "python test_pipe_dnd.py --phase test --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --save_dir {}/dndmsr --test_data {}".format(args.pretrain, args.save_dir, args.test_data)
    cmd2 = "python test_pipe_dnd.py --phase test --in_type raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --save_dir {}/dndmsr".format(args.pretrain, args.save_dir)
    cmd3 = "python test_pipe_dnd.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --save_dir {}/dndmsr --scale {}".format(args.pretrain, args.save_dir, args.scale)

elif pipeline == 1:
    # cmd1 denoise
    # cmd2 super resolution
    # cmd3 demosaic
    # DN-> SR->DM
    cmd1 = "python test_pipe_dnd.py --phase test --in_type noisy_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --save_dir {}/dnsrdm --test_data {}".format(args.pretrain, args.save_dir, args.test_data)
    cmd2 = "python test_pipe_dnd.py --phase test --in_type lr_raw --out_type raw --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --save_dir {}/dnsrdm  --scale {}".format(args.pretrain, args.save_dir, args.scale)
    cmd3 = "python test_pipe_dnd.py --phase test --in_type raw --out_type linrgb --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --save_dir {}/dnsrdm".format(args.pretrain, args.save_dir)
elif pipeline == 2:
    # cmd1 demosaic
    # cmd2 denoise
    # cmd3 super resolution
    # DM -> DN -> SR
    cmd1 = "python test_pipe_dnd.py --phase test --in_type raw --out_type linrgb --pre_out_type raw --model resnet --pretrain {} --save_dir {}/dmdnsr --test_data {}".format(args.pretrain, args.save_dir, args.test_data)
    cmd2 = "python test_pipe_dnd.py --phase test --in_type noisy_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --save_dir {}/dmdnsr".format(args.pretrain, args.save_dir)
    cmd3 = "python test_pipe_dnd.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type noisy_rgb --pre_out_type rgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --save_dir {}/dmdnsr  --scale {}".format(args.pretrain, args.save_dir, args.scale)

elif pipeline == 3:
    # cmd1 demosaic
    # cmd2 super resolution
    # cmd3 denoise
    # DM -> SR -> DN
    cmd1 = "python test_pipe_dnd.py --phase test --in_type raw --out_type linrgb --pre_out_type raw --model resnet --pretrain {} --save_dir {}/dmsrdn --test_data {}".format(args.pretrain, args.save_dir, args.test_data)
    cmd2 = "python test_pipe_dnd.py --phase test --in_type lr_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --save_dir {}/dmsrdn   --scale {}".format(args.pretrain, args.save_dir, args.scale)
    cmd3 = "python test_pipe_dnd.py --phase test --in_type noisy_linrgb --out_type linrgb --pre_in_type noisy_rgb --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --save_dir {}/dmsrdn".format(args.pretrain, args.save_dir)

elif pipeline == 4:
    # cmd1 super resolution
    # cmd2 demosaic
    # cmd3 denoise
    # SR -> DM -> DN
    cmd1 = "python test_pipe_dnd.py --phase test --in_type lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --save_dir {}/srdmdn --test_data {}  --scale {}".format(args.pretrain, args.save_dir, args.test_data, args.scale)
    cmd2 = "python test_pipe_dnd.py --phase test --in_type raw --out_type linrgb --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --save_dir {}/srdmdn".format(args.pretrain, args.save_dir)
    cmd3 = "python test_pipe_dnd.py --phase test --in_type noisy_linrgb --out_type linrgb --pre_in_type raw --pre_out_type linrgb  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --save_dir {}/srdmdn".format(args.pretrain, args.save_dir)

elif pipeline == 5:
    # cmd1 super resolution
    # cmd2 denoise
    # cmd3 demosaic
    # SR -> DN -> DM
    cmd1 = "python test_pipe_dnd.py --phase test --in_type lr_raw --out_type raw --pre_out_type raw --model resnet --pretrain {} --save_dir {}/srdndm --test_data {}  --scale {}".format(args.pretrain, args.save_dir, args.test_data, args.scale)
    cmd2 = "python test_pipe_dnd.py --phase test --in_type noisy_raw --out_type raw --pre_in_type lr_raw --pre_out_type raw  --model resnet --pre_model resnet --intermediate 1 --pretrain {} --save_dir {}/srdndm".format(args.pretrain, args.save_dir)
    cmd3 = "python test_pipe_dnd.py --phase test --in_type raw --out_type linrgb --pre_in_type noisy_raw --pre_out_type raw  --model resnet --pre_model resnet  --intermediate 1 --pretrain {} --save_dir {}/srdndm".format(args.pretrain, args.save_dir)

return_value = os.system(cmd1)
# return_value = os.system(cmd2)
# return_value = os.system(cmd3)



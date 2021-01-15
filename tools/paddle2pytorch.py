import collections
import os
import json
import paddle.fluid.dygraph as D
import torch
from paddle import fluid
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Paddle2pytorch script")
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='./pretrained/HRNet_W48_C_ssld_pretrained',
        help='input paddle model path')
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./pretrained/HRNet_W48_C_ssld_pretrained.pth',
        help='output torch model path')
    args = parser.parse_args()
    return args

def build_params_map():
    """
    build params map from paddle-paddle's hrnet to pytorch's hrnet
    """
    weight_map = collections.OrderedDict()
    # first two convolution layers
    for i in (1,2):
        weight_map['conv_layer1_%d._conv.weight'%i] = 'conv%d.weight'%i
        weight_map['conv_layer1_%d._batch_norm.weight'%i] = 'bn%d.weight'%i
        weight_map['conv_layer1_%d._batch_norm.bias'%i] = 'bn%d.bias'%i
        weight_map['conv_layer1_%d._batch_norm._mean'%i] = 'bn%d.running_mean'%i
        weight_map['conv_layer1_%d._batch_norm._variance'%i] = 'bn%d.running_var'%i

    # layer 1
    for layer in (1,2,3,4): # 1,2,3,4
        for conv in (1,2,3): # 1,2,3
            weight_map['la1.bb_layer2_{}.conv{}._conv.weight'.format(layer, conv)] = \
            'layer1.{}.conv{}.weight'.format(layer-1, conv)
            weight_map['la1.bb_layer2_{}.conv{}._batch_norm.weight'.format(layer, conv)] = \
            'layer1.{}.bn{}.weight'.format(layer-1, conv)
            weight_map['la1.bb_layer2_{}.conv{}._batch_norm.bias'.format(layer, conv)] = \
            'layer1.{}.bn{}.bias'.format(layer-1, conv)
            weight_map['la1.bb_layer2_{}.conv{}._batch_norm._mean'.format(layer, conv)] = \
            'layer1.{}.bn{}.running_mean'.format(layer-1, conv)
            weight_map['la1.bb_layer2_{}.conv{}._batch_norm._variance'.format(layer, conv)] = \
            'layer1.{}.bn{}.running_var'.format(layer-1, conv)
        if layer == 1:
            weight_map['la1.bb_layer2_{}.conv_down._conv.weight'.format(layer)] = \
            'layer1.{}.downsample.0.weight'.format(layer-1)
            weight_map['la1.bb_layer2_{}.conv_down._batch_norm.weight'.format(layer)] = \
            'layer1.{}.downsample.1.weight'.format(layer-1)
            weight_map['la1.bb_layer2_{}.conv_down._batch_norm.bias'.format(layer)] = \
            'layer1.{}.downsample.1.bias'.format(layer-1)
            weight_map['la1.bb_layer2_{}.conv_down._batch_norm._mean'.format(layer)] = \
            'layer1.{}.downsample.1.running_mean'.format(layer-1)
            weight_map['la1.bb_layer2_{}.conv_down._batch_norm._variance'.format(layer)] = \
            'layer1.{}.downsample.1.running_var'.format(layer-1)

    # layer1 transition
    for trans in (1,2): # 1,2
        if trans == 1:
            weight_map['tr1.transition_tr1_layer_{}._conv.weight'.format(trans)] = \
                'transition1.0.0.weight'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm.weight'.format(trans)] = \
                'transition1.0.1.weight'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm.bias'.format(trans)] = \
                'transition1.0.1.bias'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm._mean'.format(trans)] = \
                'transition1.0.1.running_mean'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm._variance'.format(trans)] = \
                'transition1.0.1.running_var'
        elif trans == 2:
            weight_map['tr1.transition_tr1_layer_{}._conv.weight'.format(trans)] = \
                'transition1.1.0.0.weight'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm.weight'.format(trans)] = \
                'transition1.1.0.1.weight'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm.bias'.format(trans)] = \
                'transition1.1.0.1.bias'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm._mean'.format(trans)] = \
                'transition1.1.0.1.running_mean'
            weight_map['tr1.transition_tr1_layer_{}._batch_norm._variance'.format(trans)] = \
                'transition1.1.0.1.running_var'
    
    # fuse_pattern_conv_2 = "st[0-9].stage_st([0-9])_([0-9]).fuse_func.residual_st[0-9]_[0-9]_layer_([0-9])_([0-9])._conv.weight"
    fuse_dict = collections.OrderedDict({
        #0
        '_1_2._conv.weight':'0.1.0.weight',
        '_1_2._batch_norm.weight':'0.1.1.weight',
        '_1_2._batch_norm.bias':'0.1.1.bias',
        '_1_2._batch_norm._mean':'0.1.1.running_mean',
        '_1_2._batch_norm._variance':'0.1.1.running_var',
        #1
        '_2_1_1._conv.weight':'1.0.0.0.weight',
        '_2_1_1._batch_norm.weight':'1.0.0.1.weight',
        '_2_1_1._batch_norm.bias':'1.0.0.1.bias',
        '_2_1_1._batch_norm._mean':'1.0.0.1.running_mean',
        '_2_1_1._batch_norm._variance':'1.0.0.1.running_var',

        # --------------
        #2
        '_1_3._conv.weight':'0.2.0.weight',
        '_1_3._batch_norm.weight':'0.2.1.weight',
        '_1_3._batch_norm.bias':'0.2.1.bias',
        '_1_3._batch_norm._mean':'0.2.1.running_mean',
        '_1_3._batch_norm._variance':'0.2.1.running_var',
        #3
        '_2_3._conv.weight':'1.2.0.weight',
        '_2_3._batch_norm.weight':'1.2.1.weight',
        '_2_3._batch_norm.bias':'1.2.1.bias',
        '_2_3._batch_norm._mean':'1.2.1.running_mean',
        '_2_3._batch_norm._variance':'1.2.1.running_var',
        
        #4
        '_3_1_1._conv.weight':'2.0.0.0.weight',
        '_3_1_1._batch_norm.weight':'2.0.0.1.weight',
        '_3_1_1._batch_norm.bias':'2.0.0.1.bias',
        '_3_1_1._batch_norm._mean':'2.0.0.1.running_mean',
        '_3_1_1._batch_norm._variance':'2.0.0.1.running_var',
        
        #5
        '_3_1_2._conv.weight':'2.0.1.0.weight',
        '_3_1_2._batch_norm.weight':'2.0.1.1.weight',
        '_3_1_2._batch_norm.bias':'2.0.1.1.bias',
        '_3_1_2._batch_norm._mean':'2.0.1.1.running_mean',
        '_3_1_2._batch_norm._variance':'2.0.1.1.running_var',

        #6
        '_3_2_1._conv.weight':'2.1.0.0.weight',
        '_3_2_1._batch_norm.weight':'2.1.0.1.weight',
        '_3_2_1._batch_norm.bias':'2.1.0.1.bias',
        '_3_2_1._batch_norm._mean':'2.1.0.1.running_mean',
        '_3_2_1._batch_norm._variance':'2.1.0.1.running_var',

        #7
        '_1_4._conv.weight':'0.3.0.weight',
        '_1_4._batch_norm.weight':'0.3.1.weight',
        '_1_4._batch_norm.bias':'0.3.1.bias',
        '_1_4._batch_norm._mean':'0.3.1.running_mean',
        '_1_4._batch_norm._variance':'0.3.1.running_var',

        #8
        '_2_4._conv.weight':'1.3.0.weight',
        '_2_4._batch_norm.weight':'1.3.1.weight',
        '_2_4._batch_norm.bias':'1.3.1.bias',
        '_2_4._batch_norm._mean':'1.3.1.running_mean',
        '_2_4._batch_norm._variance':'1.3.1.running_var',

        #9
        '_3_4._conv.weight':'2.3.0.weight',
        '_3_4._batch_norm.weight':'2.3.1.weight',
        '_3_4._batch_norm.bias':'2.3.1.bias',
        '_3_4._batch_norm._mean':'2.3.1.running_mean',
        '_3_4._batch_norm._variance':'2.3.1.running_var',

        #10
        '_4_1_1._conv.weight':'3.0.0.0.weight',
        '_4_1_1._batch_norm.weight':'3.0.0.1.weight',
        '_4_1_1._batch_norm.bias':'3.0.0.1.bias',
        '_4_1_1._batch_norm._mean':'3.0.0.1.running_mean',
        '_4_1_1._batch_norm._variance':'3.0.0.1.running_var',

        #11
        '_4_1_2._conv.weight':'3.0.1.0.weight',
        '_4_1_2._batch_norm.weight':'3.0.1.1.weight',
        '_4_1_2._batch_norm.bias':'3.0.1.1.bias',
        '_4_1_2._batch_norm._mean':'3.0.1.1.running_mean',
        '_4_1_2._batch_norm._variance':'3.0.1.1.running_var',

        #12
        '_4_1_3._conv.weight':'3.0.2.0.weight',
        '_4_1_3._batch_norm.weight':'3.0.2.1.weight',
        '_4_1_3._batch_norm.bias':'3.0.2.1.bias',
        '_4_1_3._batch_norm._mean':'3.0.2.1.running_mean',
        '_4_1_3._batch_norm._variance':'3.0.2.1.running_var',

        #13
        '_4_2_1._conv.weight':'3.1.0.0.weight',
        '_4_2_1._batch_norm.weight':'3.1.0.1.weight',
        '_4_2_1._batch_norm.bias':'3.1.0.1.bias',
        '_4_2_1._batch_norm._mean':'3.1.0.1.running_mean',
        '_4_2_1._batch_norm._variance':'3.1.0.1.running_var',

        #14
        '_4_2_2._conv.weight':'3.1.1.0.weight',
        '_4_2_2._batch_norm.weight':'3.1.1.1.weight',
        '_4_2_2._batch_norm.bias':'3.1.1.1.bias',
        '_4_2_2._batch_norm._mean':'3.1.1.1.running_mean',
        '_4_2_2._batch_norm._variance':'3.1.1.1.running_var',

        #15
        '_4_3_1._conv.weight':'3.2.0.0.weight',
        '_4_3_1._batch_norm.weight':'3.2.0.1.weight',
        '_4_3_1._batch_norm.bias':'3.2.0.1.bias',
        '_4_3_1._batch_norm._mean':'3.2.0.1.running_mean',
        '_4_3_1._batch_norm._variance':'3.2.0.1.running_var',

    })

    # stages 2-4
    for stage in (2,3,4):
        if stage == 2:
            s_affix = 1
            for branch in (1,2):
                for b_affix in (1,2,3,4):
                    for conv in (1,2):
                        weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._conv.weight'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.conv{}.weight'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                        weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm.weight'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.bn{}.weight'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                        weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm.bias'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.bn{}.bias'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                        weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm._mean'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.bn{}.running_mean'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                        weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm._variance'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.bn{}.running_var'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
            # fuse for stage 2
            fuse_no = (0,1)
            fuse_dict_list = []
            for no in fuse_no:
                fuse_dict_list += list(fuse_dict.items())[no*5:(no+1)*5]
            print(fuse_dict_list)
            for key, value in fuse_dict_list:
                print("fuse dict", key, value)
                weight_map['st{}.stage_st{}_{}.fuse_func.residual_st{}_{}_layer{}'. \
                    format(stage, stage, s_affix, stage, s_affix, key)] = \
                        'stage{}.{}.fuse_layers.{}'.format(stage, s_affix-1, value)
            # transition for stage 2
            weight_map['tr2.transition_tr2_layer_3._conv.weight'] = 'transition2.2.0.0.weight'
            weight_map['tr2.transition_tr2_layer_3._batch_norm.weight'] = 'transition2.2.0.1.weight'
            weight_map['tr2.transition_tr2_layer_3._batch_norm.bias'] = 'transition2.2.0.1.bias'
            weight_map['tr2.transition_tr2_layer_3._batch_norm._mean'] = 'transition2.2.0.1.running_mean'
            weight_map['tr2.transition_tr2_layer_3._batch_norm._variance'] = 'transition2.2.0.1.running_var'

        if stage == 3:
            for s_affix in (1,2,3,4):
                for branch in (1,2,3):
                    for b_affix in (1,2,3,4):
                        for conv in (1,2):
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._conv.weight'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.conv{}.weight'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm.weight'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.weight'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm.bias'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.bias'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm._mean'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.running_mean'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm._variance'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.running_var'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                # fuse layer for stage_affix
                fuse_no = (0,2,1,3,4,5,6)
                fuse_dict_list = []
                for no in fuse_no:
                    fuse_dict_list += list(fuse_dict.items())[no*5:(no+1)*5]
                for key, value in fuse_dict_list:
                    print("fuse dict", key, value)
                    weight_map['st{}.stage_st{}_{}.fuse_func.residual_st{}_{}_layer{}'. \
                        format(stage, stage, s_affix, stage, s_affix, key)] = \
                            'stage{}.{}.fuse_layers.{}'.format(stage, s_affix-1, value)
                
            # transition layer for stage_affix
            weight_map['tr3.transition_tr3_layer_4._conv.weight'] = 'transition3.3.0.0.weight'
            weight_map['tr3.transition_tr3_layer_4._batch_norm.weight'] = 'transition3.3.0.1.weight'
            weight_map['tr3.transition_tr3_layer_4._batch_norm.bias'] = 'transition3.3.0.1.bias'
            weight_map['tr3.transition_tr3_layer_4._batch_norm._mean'] = 'transition3.3.0.1.running_mean'
            weight_map['tr3.transition_tr3_layer_4._batch_norm._variance'] = 'transition3.3.0.1.running_var'

        if stage==4:
            for s_affix in (1,2,3):
                for branch in (1,2,3,4):
                    for b_affix in (1,2,3,4):
                        for conv in (1,2):
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._conv.weight'. \
                            format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                'stage{}.{}.branches.{}.{}.conv{}.weight'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm.weight'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.weight'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm.bias'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.bias'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm._mean'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.running_mean'.format(stage, s_affix-1, branch-1, b_affix-1, conv)
                            weight_map['st{}.stage_st{}_{}.branches_func.bb_st{}_{}_branch_layer_{}_{}.conv{}._batch_norm._variance'. \
                                format(stage, stage, s_affix, stage, s_affix, branch, b_affix, conv)] = \
                                    'stage{}.{}.branches.{}.{}.bn{}.running_var'.format(stage, s_affix-1, branch-1, b_affix-1, conv)

                # fuse layer for stage_affix
                    fuse_no = (0,2,7,1,3,8,4,5,6,9,10,11,12,13,14,15)
                    fuse_dict_list = []
                    for no in fuse_no:
                        fuse_dict_list += list(fuse_dict.items())[no*5:(no+1)*5]
                    for key, value in fuse_dict_list:
                        print("fuse dict", key, value)
                        weight_map['st{}.stage_st{}_{}.fuse_func.residual_st{}_{}_layer{}'. \
                            format(stage, stage, s_affix, stage, s_affix, key)] = \
                                'stage{}.{}.fuse_layers.{}'.format(stage, s_affix-1, value)

    # cls layer
    for layer in (1,2,3,4):
        for conv in (1,2,3):
            weight_map['last_cls.conv_cls_head_conv_{}.conv{}._conv.weight'.format(layer, conv)] = \
                    'incre_modules.{}.0.conv{}.weight'.format(layer-1, conv)
            weight_map['last_cls.conv_cls_head_conv_{}.conv{}._batch_norm.weight'.format(layer, conv)] = \
                    'incre_modules.{}.0.bn{}.weight'.format(layer-1, conv)
            weight_map['last_cls.conv_cls_head_conv_{}.conv{}._batch_norm.bias'.format(layer, conv)] = \
                    'incre_modules.{}.0.bn{}.bias'.format(layer-1, conv)
            weight_map['last_cls.conv_cls_head_conv_{}.conv{}._batch_norm._mean'.format(layer, conv)] = \
                    'incre_modules.{}.0.bn{}.running_mean'.format(layer-1, conv)
            weight_map['last_cls.conv_cls_head_conv_{}.conv{}._batch_norm._variance'.format(layer, conv)] = \
                    'incre_modules.{}.0.bn{}.running_var'.format(layer-1, conv)
        # downsample
        weight_map['last_cls.conv_cls_head_conv_{}.conv_down._conv.weight'.format(layer)] = \
                    'incre_modules.{}.0.downsample.0.weight'.format(layer-1)
        weight_map['last_cls.conv_cls_head_conv_{}.conv_down._batch_norm.weight'.format(layer)] = \
                'incre_modules.{}.0.downsample.1.weight'.format(layer-1)
        weight_map['last_cls.conv_cls_head_conv_{}.conv_down._batch_norm.bias'.format(layer)] = \
                'incre_modules.{}.0.downsample.1.bias'.format(layer-1)
        weight_map['last_cls.conv_cls_head_conv_{}.conv_down._batch_norm._mean'.format(layer)] = \
                'incre_modules.{}.0.downsample.1.running_mean'.format(layer-1)
        weight_map['last_cls.conv_cls_head_conv_{}.conv_down._batch_norm._variance'.format(layer)] = \
                'incre_modules.{}.0.downsample.1.running_var'.format(layer-1)

    # pre final layer
    for layer in (1,2,3,4):
        weight_map['cls_head_add{}._conv.weight'.format(layer)] = \
                    'downsamp_modules.{}.0.weight'.format(layer-1)
        weight_map['cls_head_add{}._batch_norm.weight'.format(layer)] = \
                'downsamp_modules.{}.1.weight'.format(layer-1)
        weight_map['cls_head_add{}._batch_norm.bias'.format(layer)] = \
                'downsamp_modules.{}.1.bias'.format(layer-1)
        weight_map['cls_head_add{}._batch_norm._mean'.format(layer)] = \
                'downsamp_modules.{}.1.running_mean'.format(layer-1)
        weight_map['cls_head_add{}._batch_norm._variance'.format(layer)] = \
                'downsamp_modules.{}.1.running_var'.format(layer-1)
    
    # final layer
    weight_map['conv_last._conv.weight'] = 'final_layer.0.weight'
    weight_map['conv_last._batch_norm.weight'] = 'final_layer.1.weight'
    weight_map['conv_last._batch_norm.bias'] = 'final_layer.1.bias'
    weight_map['conv_last._batch_norm._mean'] = 'final_layer.1.running_mean'
    weight_map['conv_last._batch_norm._variance'] = 'final_layer.1.running_var'

    weight_map['out.weight'] = 'classifier.weight'
    weight_map['out.bias'] = 'classifier.bias'

    
    print("Generated %d dict item"%len(weight_map))
    return weight_map

def extract_and_convert(input_path, output_path):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    state_dict = collections.OrderedDict()
    # state = fluid.io.load_program_state(path + ".pdparams")
    weight_map = build_params_map()
    checkpoint = torch.load(os.path.join('./pretrained/hrnetv2_w18-00eb2006.pth'), map_location=None)
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph(input_path)
    count = 0
    for weight_name, weight_value in paddle_paddle_params.items():
        if weight_name == 'out.weight':
            weight_value = weight_value.transpose()
            print(weight_value.shape)
        if weight_map[weight_name] in checkpoint.keys():
            count += 1
            print("%d keys same as pytorch parameters"%count)
            state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        
    # print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, output_path)
    
    # for weight_name, weight_value in checkpoint.items():
    #     print("weight_name, weight value", type(weight_name), type(weight_value))

def test_two_pytorch(input_dir):
    checkpoint1 = torch.load(os.path.join(input_dir, 'hrnetv2_w18-00eb2006.pth'), map_location=None)
    checkpoint2 = torch.load(os.path.join(input_dir, 'pytorch_model.pth'), map_location=None)
    print(len(checkpoint1))
    print(len(checkpoint2))
    print(checkpoint1.keys())
    print(checkpoint2.keys())
    # for key, value in checkpoint1.items():
    #     print(type(key), type(value))
    # print("##############")
    # for key, value in checkpoint2.items():
    #     print(type(key), type(value))

if __name__ == "__main__":
    args = parse_args()
    extract_and_convert(args.input, args.output)
    # test_two_pytorch("../pretrained")
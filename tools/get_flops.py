import argparse
from functools import partial

import mmengine
import mmengine.dataset
import torch
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--infos', help='Infos file with annotations')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM2',
        help='choose camera type to inference')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1248, 384],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='image',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def input_constructor(model,
                      input_key,
                      input_shape,
                      ann_file=None,
                      cam_type=None):
    try:
        batch = torch.ones(()).new_empty(
            (1, *input_shape),
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device)
    except StopIteration:
        # Avoid StopIteration for models which have no parameters,
        # like `nn.Relu()`, `nn.AvgPool2d`, etc.
        batch = torch.ones(()).new_empty((1, *input_shape))

    metainfo = None
    if ann_file is not None:
        data_list = mmengine.load(ann_file)['data_list']
        data_info = data_list[0]
        metainfo = dict(img_shape=input_shape, **data_info['images'][cam_type])

    inputs = {}
    inputs[input_key] = batch
    data_samples = [Det3DDataSample(metainfo=metainfo)]
    input = dict(inputs=inputs, data_samples=data_samples)

    return input


def main():
    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
        input_key = 'points'
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
        input_key = 'imgs'
    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    flops, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=partial(
            input_constructor,
            model,
            input_key,
            ann_file=args.infos,
            cam_type=args.cam_type))
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

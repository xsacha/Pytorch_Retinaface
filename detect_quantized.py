from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils import mkldnn
from torch.cuda.amp import autocast

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Retinaface_detnas60.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--mkl', action="store_true", default=False, help='Use mkldnn inference')
parser.add_argument('--amp', action="store_true", default=False, help='Use AMP inference')
parser.add_argument('--mobile', action="store_true", default=False, help='Optimise for mobile')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def test(model, device, args):
    resize = 1

    # testing begin
    for i in range(100):
        image_path = "./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= 127.5#(104, 117, 123)
        img /= 127.5
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        if args.mkl:
            img = img.to_mkldnn()
        if args.amp:
            img = img.half()
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 4, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 4, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 4, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 4, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 4, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = RetinaFace(phase="test", net="detnas")
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    #test(net, device, args)
    #torch.backends.quantized.engine='fbgemm'

    #net.fuse_model()
    # set quantization config for server (x86)
    #net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
     
    torch.quantization.fuse_modules(net, [
        ['body.0.conv.0.interstellar1a_shufflenet_5x5_branch2a',   'body.0.conv.0.bn1a_shufflenet_5x5_branch2a', 'body.0.conv.0.relu1a_shufflenet_5x5_branch2a'],
        ['body.0.conv.1.interstellar1a_shufflenet_5x5_branch2b_s', 'body.0.conv.1.interstellar1a_shufflenet_5x5_branch2b_s_bn'],
        ['body.0.conv.1.interstellar1a_shufflenet_5x5_branch2b',   'body.0.conv.1.bn1a_shufflenet_5x5_branch2b'],
        ['body.0.proj_conv.interstellar1a_proj_s',                 'body.0.proj_conv.interstellar1a_proj_s_bn'],
        ['body.0.proj_conv.interstellar1a_proj',                   'body.0.proj_conv.bn1a_proj', 'body.0.proj_conv.relu1a_proj'],
        ['body.1.conv.0.interstellar1b_xception_3x3_branch2a_s',   'body.1.conv.0.interstellar1b_xception_3x3_branch2a_s_bn'],
        ['body.1.conv.0.interstellar1b_xception_3x3_branch2a',     'body.1.conv.0.bn1b_xception_3x3_branch2a', 'body.1.conv.0.relu1b_xception_3x3_branch2a'],
        ['body.1.conv.1.interstellar1b_xception_3x3_branch2b_s',   'body.1.conv.1.interstellar1b_xception_3x3_branch2b_s_bn'],
        ['body.1.conv.1.interstellar1b_xception_3x3_branch2b',     'body.1.conv.1.bn1b_xception_3x3_branch2b', 'body.1.conv.1.relu1b_xception_3x3_branch2b'],
        ['body.1.conv.2.interstellar1b_xception_3x3_branch2c_s',   'body.1.conv.2.interstellar1b_xception_3x3_branch2c_s_bn'],
        ['body.1.conv.2.interstellar1b_xception_3x3_branch2c',     'body.1.conv.2.bn1b_xception_3x3_branch2c'],
        ['body.2.conv.0.interstellar1c_shufflenet_3x3_branch2a',   'body.2.conv.0.bn1c_shufflenet_3x3_branch2a', 'body.2.conv.0.relu1c_shufflenet_3x3_branch2a'],
        ['body.2.conv.1.interstellar1c_shufflenet_3x3_branch2b_s', 'body.2.conv.1.interstellar1c_shufflenet_3x3_branch2b_s_bn'],
        ['body.2.conv.1.interstellar1c_shufflenet_3x3_branch2b',   'body.2.conv.1.bn1c_shufflenet_3x3_branch2b'],
        ['body.3.conv.0.interstellar1d_shufflenet_3x3_branch2a',   'body.3.conv.0.bn1d_shufflenet_3x3_branch2a', 'body.3.conv.0.relu1d_shufflenet_3x3_branch2a'],
        ['body.3.conv.1.interstellar1d_shufflenet_3x3_branch2b_s', 'body.3.conv.1.interstellar1d_shufflenet_3x3_branch2b_s_bn'],
        ['body.3.conv.1.interstellar1d_shufflenet_3x3_branch2b',   'body.3.conv.1.bn1d_shufflenet_3x3_branch2b'],
        ['body.4.conv.0.interstellar2a_shufflenet_7x7_branch2a',   'body.4.conv.0.bn2a_shufflenet_7x7_branch2a', 'body.4.conv.0.relu2a_shufflenet_7x7_branch2a'],
        ['body.4.conv.1.interstellar2a_shufflenet_7x7_branch2b_s', 'body.4.conv.1.interstellar2a_shufflenet_7x7_branch2b_s_bn'],
        ['body.4.conv.1.interstellar2a_shufflenet_7x7_branch2b',   'body.4.conv.1.bn2a_shufflenet_7x7_branch2b'],
        ['body.4.proj_conv.interstellar2a_proj_s',                 'body.4.proj_conv.interstellar2a_proj_s_bn'],
        ['body.4.proj_conv.interstellar2a_proj',                   'body.4.proj_conv.bn2a_proj', 'body.4.proj_conv.relu2a_proj'],
        ['body.5.conv.0.interstellar2b_xception_3x3_branch2a_s',   'body.5.conv.0.interstellar2b_xception_3x3_branch2a_s_bn'],
        ['body.5.conv.0.interstellar2b_xception_3x3_branch2a',     'body.5.conv.0.bn2b_xception_3x3_branch2a', 'body.5.conv.0.relu2b_xception_3x3_branch2a'],
        ['body.5.conv.1.interstellar2b_xception_3x3_branch2b_s',   'body.5.conv.1.interstellar2b_xception_3x3_branch2b_s_bn'],
        ['body.5.conv.1.interstellar2b_xception_3x3_branch2b',     'body.5.conv.1.bn2b_xception_3x3_branch2b', 'body.5.conv.1.relu2b_xception_3x3_branch2b'],
        ['body.5.conv.2.interstellar2b_xception_3x3_branch2c_s',   'body.5.conv.2.interstellar2b_xception_3x3_branch2c_s_bn'],
        ['body.5.conv.2.interstellar2b_xception_3x3_branch2c',     'body.5.conv.2.bn2b_xception_3x3_branch2c'],
        ['body.6.conv.0.interstellar2c_xception_3x3_branch2a_s',   'body.6.conv.0.interstellar2c_xception_3x3_branch2a_s_bn'],
        ['body.6.conv.0.interstellar2c_xception_3x3_branch2a',     'body.6.conv.0.bn2c_xception_3x3_branch2a', 'body.6.conv.0.relu2c_xception_3x3_branch2a'],
        ['body.6.conv.1.interstellar2c_xception_3x3_branch2b_s',   'body.6.conv.1.interstellar2c_xception_3x3_branch2b_s_bn'],
        ['body.6.conv.1.interstellar2c_xception_3x3_branch2b',     'body.6.conv.1.bn2c_xception_3x3_branch2b', 'body.6.conv.1.relu2c_xception_3x3_branch2b'],
        ['body.6.conv.2.interstellar2c_xception_3x3_branch2c_s',   'body.6.conv.2.interstellar2c_xception_3x3_branch2c_s_bn'],
        ['body.6.conv.2.interstellar2c_xception_3x3_branch2c',     'body.6.conv.2.bn2c_xception_3x3_branch2c'],
        ['body.7.conv.0.interstellar2d_xception_3x3_branch2a_s',   'body.7.conv.0.interstellar2d_xception_3x3_branch2a_s_bn'],
        ['body.7.conv.0.interstellar2d_xception_3x3_branch2a',     'body.7.conv.0.bn2d_xception_3x3_branch2a', 'body.7.conv.0.relu2d_xception_3x3_branch2a'],
        ['body.7.conv.1.interstellar2d_xception_3x3_branch2b_s',   'body.7.conv.1.interstellar2d_xception_3x3_branch2b_s_bn'],
        ['body.7.conv.1.interstellar2d_xception_3x3_branch2b',     'body.7.conv.1.bn2d_xception_3x3_branch2b', 'body.7.conv.1.relu2d_xception_3x3_branch2b'],
        ['body.7.conv.2.interstellar2d_xception_3x3_branch2c_s',   'body.7.conv.2.interstellar2d_xception_3x3_branch2c_s_bn'],
        ['body.7.conv.2.interstellar2d_xception_3x3_branch2c',     'body.7.conv.2.bn2d_xception_3x3_branch2c'],
        ['body.8.conv.0.interstellar3a_shufflenet_7x7_branch2a',   'body.8.conv.0.bn3a_shufflenet_7x7_branch2a', 'body.8.conv.0.relu3a_shufflenet_7x7_branch2a'],
        ['body.8.conv.1.interstellar3a_shufflenet_7x7_branch2b_s', 'body.8.conv.1.interstellar3a_shufflenet_7x7_branch2b_s_bn'],
        ['body.8.conv.1.interstellar3a_shufflenet_7x7_branch2b',   'body.8.conv.1.bn3a_shufflenet_7x7_branch2b'],
        ['body.8.proj_conv.interstellar3a_proj_s',                 'body.8.proj_conv.interstellar3a_proj_s_bn'],
        ['body.8.proj_conv.interstellar3a_proj',                   'body.8.proj_conv.bn3a_proj', 'body.8.proj_conv.relu3a_proj'],
        ['body.9.conv.0.interstellar3b_xception_3x3_branch2a_s',   'body.9.conv.0.interstellar3b_xception_3x3_branch2a_s_bn'],
        ['body.9.conv.0.interstellar3b_xception_3x3_branch2a',     'body.9.conv.0.bn3b_xception_3x3_branch2a', 'body.9.conv.0.relu3b_xception_3x3_branch2a'],
        ['body.9.conv.1.interstellar3b_xception_3x3_branch2b_s',   'body.9.conv.1.interstellar3b_xception_3x3_branch2b_s_bn'],
        ['body.9.conv.1.interstellar3b_xception_3x3_branch2b',     'body.9.conv.1.bn3b_xception_3x3_branch2b', 'body.9.conv.1.relu3b_xception_3x3_branch2b'],
        ['body.9.conv.2.interstellar3b_xception_3x3_branch2c_s',   'body.9.conv.2.interstellar3b_xception_3x3_branch2c_s_bn'],
        ['body.9.conv.2.interstellar3b_xception_3x3_branch2c',     'body.9.conv.2.bn3b_xception_3x3_branch2c'],
        ['body.10.conv.0.interstellar3c_xception_3x3_branch2a_s',   'body.10.conv.0.interstellar3c_xception_3x3_branch2a_s_bn'],
        ['body.10.conv.0.interstellar3c_xception_3x3_branch2a',     'body.10.conv.0.bn3c_xception_3x3_branch2a', 'body.10.conv.0.relu3c_xception_3x3_branch2a'],
        ['body.10.conv.1.interstellar3c_xception_3x3_branch2b_s',   'body.10.conv.1.interstellar3c_xception_3x3_branch2b_s_bn'],
        ['body.10.conv.1.interstellar3c_xception_3x3_branch2b',     'body.10.conv.1.bn3c_xception_3x3_branch2b', 'body.10.conv.1.relu3c_xception_3x3_branch2b'],
        ['body.10.conv.2.interstellar3c_xception_3x3_branch2c_s',   'body.10.conv.2.interstellar3c_xception_3x3_branch2c_s_bn'],
        ['body.10.conv.2.interstellar3c_xception_3x3_branch2c',     'body.10.conv.2.bn3c_xception_3x3_branch2c'],
        ['body.11.conv.0.interstellar3d_xception_3x3_branch2a_s',   'body.11.conv.0.interstellar3d_xception_3x3_branch2a_s_bn'],
        ['body.11.conv.0.interstellar3d_xception_3x3_branch2a',     'body.11.conv.0.bn3d_xception_3x3_branch2a', 'body.11.conv.0.relu3d_xception_3x3_branch2a'],
        ['body.11.conv.1.interstellar3d_xception_3x3_branch2b_s',   'body.11.conv.1.interstellar3d_xception_3x3_branch2b_s_bn'],
        ['body.11.conv.1.interstellar3d_xception_3x3_branch2b',     'body.11.conv.1.bn3d_xception_3x3_branch2b', 'body.11.conv.1.relu3d_xception_3x3_branch2b'],
        ['body.11.conv.2.interstellar3d_xception_3x3_branch2c_s',   'body.11.conv.2.interstellar3d_xception_3x3_branch2c_s_bn'],
        ['body.11.conv.2.interstellar3d_xception_3x3_branch2c',     'body.11.conv.2.bn3d_xception_3x3_branch2c'],
        ['body.12.conv.0.interstellar3e_xception_3x3_branch2a_s',   'body.12.conv.0.interstellar3e_xception_3x3_branch2a_s_bn'],
        ['body.12.conv.0.interstellar3e_xception_3x3_branch2a',     'body.12.conv.0.bn3e_xception_3x3_branch2a', 'body.12.conv.0.relu3e_xception_3x3_branch2a'],
        ['body.12.conv.1.interstellar3e_xception_3x3_branch2b_s',   'body.12.conv.1.interstellar3e_xception_3x3_branch2b_s_bn'],
        ['body.12.conv.1.interstellar3e_xception_3x3_branch2b',     'body.12.conv.1.bn3e_xception_3x3_branch2b', 'body.12.conv.1.relu3e_xception_3x3_branch2b'],
        ['body.12.conv.2.interstellar3e_xception_3x3_branch2c_s',   'body.12.conv.2.interstellar3e_xception_3x3_branch2c_s_bn'],
        ['body.12.conv.2.interstellar3e_xception_3x3_branch2c',     'body.12.conv.2.bn3e_xception_3x3_branch2c'],
        ['body.13.conv.0.interstellar3f_shufflenet_7x7_branch2a',   'body.13.conv.0.bn3f_shufflenet_7x7_branch2a', 'body.13.conv.0.relu3f_shufflenet_7x7_branch2a'],
        ['body.13.conv.1.interstellar3f_shufflenet_7x7_branch2b_s', 'body.13.conv.1.interstellar3f_shufflenet_7x7_branch2b_s_bn'],
        ['body.13.conv.1.interstellar3f_shufflenet_7x7_branch2b',   'body.13.conv.1.bn3f_shufflenet_7x7_branch2b'],
        ['body.14.conv.0.interstellar3g_shufflenet_7x7_branch2a',   'body.14.conv.0.bn3g_shufflenet_7x7_branch2a', 'body.14.conv.0.relu3g_shufflenet_7x7_branch2a'],
        ['body.14.conv.1.interstellar3g_shufflenet_7x7_branch2b_s', 'body.14.conv.1.interstellar3g_shufflenet_7x7_branch2b_s_bn'],
        ['body.14.conv.1.interstellar3g_shufflenet_7x7_branch2b',   'body.14.conv.1.bn3g_shufflenet_7x7_branch2b'],
        ['body.15.conv.0.interstellar3h_shufflenet_3x3_branch2a',   'body.15.conv.0.bn3h_shufflenet_3x3_branch2a', 'body.15.conv.0.relu3h_shufflenet_3x3_branch2a'],
        ['body.15.conv.1.interstellar3h_shufflenet_3x3_branch2b_s', 'body.15.conv.1.interstellar3h_shufflenet_3x3_branch2b_s_bn'],
        ['body.15.conv.1.interstellar3h_shufflenet_3x3_branch2b',   'body.15.conv.1.bn3h_shufflenet_3x3_branch2b'],
        ['body.16.conv.0.interstellar4a_shufflenet_7x7_branch2a',   'body.16.conv.0.bn4a_shufflenet_7x7_branch2a', 'body.16.conv.0.relu4a_shufflenet_7x7_branch2a'],
        ['body.16.conv.1.interstellar4a_shufflenet_7x7_branch2b_s', 'body.16.conv.1.interstellar4a_shufflenet_7x7_branch2b_s_bn'],
        ['body.16.conv.1.interstellar4a_shufflenet_7x7_branch2b',   'body.16.conv.1.bn4a_shufflenet_7x7_branch2b'],
        ['body.16.proj_conv.interstellar4a_proj_s',                 'body.16.proj_conv.interstellar4a_proj_s_bn'],
        ['body.16.proj_conv.interstellar4a_proj',                   'body.16.proj_conv.bn4a_proj', 'body.16.proj_conv.relu4a_proj'],
        ['fpn.output1.0', 'fpn.output1.1', 'fpn.output1.2'],
        ['fpn.output2.0', 'fpn.output2.1', 'fpn.output2.2'],
        ['fpn.output3.0', 'fpn.output3.1', 'fpn.output3.2'],
        ['fpn.merge1.0', 'fpn.merge1.1', 'fpn.merge1.2'],
        ['fpn.merge2.0', 'fpn.merge2.1', 'fpn.merge2.2'],
        ['ssh1.conv3X3.0',   'ssh1.conv3X3.1'],
        ['ssh1.conv5X5_1.0', 'ssh1.conv5X5_1.1', 'ssh1.conv5X5_1.2'],
        ['ssh1.conv5X5_2.0', 'ssh1.conv5X5_2.1'],
        ['ssh1.conv7X7_2.0', 'ssh1.conv7X7_2.1', 'ssh1.conv7X7_2.2'],
        ['ssh1.conv7x7_3.0', 'ssh1.conv7x7_3.1'],
        ['ssh2.conv3X3.0',   'ssh2.conv3X3.1'],
        ['ssh2.conv5X5_1.0', 'ssh2.conv5X5_1.1', 'ssh2.conv5X5_1.2'],
        ['ssh2.conv5X5_2.0', 'ssh2.conv5X5_2.1'],
        ['ssh2.conv7X7_2.0', 'ssh2.conv7X7_2.1', 'ssh2.conv7X7_2.2'],
        ['ssh2.conv7x7_3.0', 'ssh2.conv7x7_3.1'],
        ['ssh3.conv3X3.0',   'ssh3.conv3X3.1'],
        ['ssh3.conv5X5_1.0', 'ssh3.conv5X5_1.1', 'ssh3.conv5X5_1.2'],
        ['ssh3.conv5X5_2.0', 'ssh3.conv5X5_2.1'],
        ['ssh3.conv7X7_2.0', 'ssh3.conv7X7_2.1', 'ssh3.conv7X7_2.2'],
        ['ssh3.conv7x7_3.0', 'ssh3.conv7x7_3.1'],
        ], inplace=True)
    
    '''
    torch.quantization.fuse_modules(net, [['body.stage1.0.0', 'body.stage1.0.1', 'body.stage1.0.2'],
                                          ['body.stage1.1.0', 'body.stage1.1.1', 'body.stage1.1.2'],
                                          ['body.stage1.1.3', 'body.stage1.1.4', 'body.stage1.1.5'],
                                          ['body.stage1.2.0', 'body.stage1.2.1', 'body.stage1.2.2'],
                                          ['body.stage1.2.3', 'body.stage1.2.4', 'body.stage1.2.5'],
                                          ['body.stage1.3.0', 'body.stage1.3.1', 'body.stage1.3.2'],
                                          ['body.stage1.3.3', 'body.stage1.3.4', 'body.stage1.3.5'],
                                          ['body.stage1.4.0', 'body.stage1.4.1', 'body.stage1.4.2'],
                                          ['body.stage1.4.3', 'body.stage1.4.4', 'body.stage1.4.5'],
                                          ['body.stage1.5.0', 'body.stage1.5.1', 'body.stage1.5.2'],
                                          ['body.stage1.5.3', 'body.stage1.5.4', 'body.stage1.5.5'],
                                          ['body.stage2.0.0', 'body.stage2.0.1', 'body.stage2.0.2'],
                                          ['body.stage2.0.3', 'body.stage2.0.4', 'body.stage2.0.5'],
                                          ['body.stage2.1.0', 'body.stage2.1.1', 'body.stage2.1.2'],
                                          ['body.stage2.1.3', 'body.stage2.1.4', 'body.stage2.1.5'],
                                          ['body.stage2.2.0', 'body.stage2.2.1', 'body.stage2.2.2'],
                                          ['body.stage2.2.3', 'body.stage2.2.4', 'body.stage2.2.5'],
                                          ['body.stage2.3.0', 'body.stage2.3.1', 'body.stage2.3.2'],
                                          ['body.stage2.3.3', 'body.stage2.3.4', 'body.stage2.3.5'],
                                          ['body.stage2.4.0', 'body.stage2.4.1', 'body.stage2.4.2'],
                                          ['body.stage2.4.3', 'body.stage2.4.4', 'body.stage2.4.5'],
                                          ['body.stage2.5.0', 'body.stage2.5.1', 'body.stage2.5.2'],
                                          ['body.stage2.5.3', 'body.stage2.5.4', 'body.stage2.5.5'],
                                          ['body.stage3.0.0', 'body.stage3.0.1', 'body.stage3.0.2'],
                                          ['body.stage3.0.3', 'body.stage3.0.4', 'body.stage3.0.5'],
                                          ['body.stage3.1.0', 'body.stage3.1.1', 'body.stage3.1.2'],
                                          ['body.stage3.1.3', 'body.stage3.1.4', 'body.stage3.1.5'],
                                          ['fpn.output1.0', 'fpn.output1.1', 'fpn.output1.2'],
                                          ['fpn.output2.0', 'fpn.output2.1', 'fpn.output2.2'],
                                          ['fpn.output3.0', 'fpn.output3.1', 'fpn.output3.2'],
                                          ['fpn.merge1.0', 'fpn.merge1.1', 'fpn.merge1.2'],
                                          ['fpn.merge2.0', 'fpn.merge2.1', 'fpn.merge2.2'],
                                          ['ssh1.conv3X3.0',   'ssh1.conv3X3.1'],
                                          ['ssh1.conv5X5_1.0', 'ssh1.conv5X5_1.1', 'ssh1.conv5X5_1.2'],
                                          ['ssh1.conv5X5_2.0', 'ssh1.conv5X5_2.1'],
                                          ['ssh1.conv7X7_2.0', 'ssh1.conv7X7_2.1', 'ssh1.conv7X7_2.2'],
                                          ['ssh1.conv7x7_3.0', 'ssh1.conv7x7_3.1'],
                                          ['ssh2.conv3X3.0',   'ssh2.conv3X3.1'],
                                          ['ssh2.conv5X5_1.0', 'ssh2.conv5X5_1.1', 'ssh2.conv5X5_1.2'],
                                          ['ssh2.conv5X5_2.0', 'ssh2.conv5X5_2.1'],
                                          ['ssh2.conv7X7_2.0', 'ssh2.conv7X7_2.1', 'ssh2.conv7X7_2.2'],
                                          ['ssh2.conv7x7_3.0', 'ssh2.conv7x7_3.1'],
                                          ['ssh3.conv3X3.0',   'ssh3.conv3X3.1'],
                                          ['ssh3.conv5X5_1.0', 'ssh3.conv5X5_1.1', 'ssh3.conv5X5_1.2'],
                                          ['ssh3.conv5X5_2.0', 'ssh3.conv5X5_2.1'],
                                          ['ssh3.conv7X7_2.0', 'ssh3.conv7X7_2.1', 'ssh3.conv7X7_2.2'],
                                          ['ssh3.conv7x7_3.0', 'ssh3.conv7x7_3.1'],
                                          
                                         ], inplace=True)
    '''
    print(net)

    # insert observers
    #torch.quantization.prepare(net, inplace=True)
    # Calibrate the model and collect statistics

    #test(net, device, args)


    # convert to quantized version
    #torch.quantization.convert(net, inplace=True)

    #net = net.to(device)
    inp = torch.randn(1, 3, 768, 1024).to(device)
    net = net.to(device)
    if args.mkl:
        net = mkldnn.to_mkldnn(net)
        inp = inp.to_mkldnn()
    if args.amp:
        #net.half()
        inp = inp.half()
    net.eval()
    with autocast(args.amp):
        net = torch.jit.trace(net, inp, check_trace=False)
    if not args.mkl:
        net = torch.jit.freeze(net)

    if args.mobile:
        torchnet = optimize_for_mobile(net)
        torchnet.save('retina-detnas_mobile.pt')
        test(torchnet, device, args)
    else:
        net.save('retina-detnas_amp.pt')
        test(net, device, args)


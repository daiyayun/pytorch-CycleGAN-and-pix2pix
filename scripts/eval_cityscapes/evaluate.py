import os
import caffe
import argparse
import numpy as np
import scipy.misc
from PIL import Image
from util import segrun, fast_hist, get_scores
from cityscapes import cityscapes

parser = argparse.ArgumentParser()
# testB
parser.add_argument("--label_dir", type=str, required=True, help="Path to the ground truth label")
# prediction labels of testA
parser.add_argument("--num_test", type=int, default=500, help="number of images to evaluate")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
parser.add_argument("--caffemodel_dir", type=str, default='./scripts/eval_cityscapes/caffemodel/', help="Where the FCN-8s caffemodel stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
# parser.add_argument("--split", type=str, default='test', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")
args = parser.parse_args()


def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = args.output_dir + 'image_outputs/'
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
    CS = cityscapes(args.label_dir)
    n_cl = len(CS.classes)
    # label_frames = CS.list_label_frames(args.split)
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(args.caffemodel_dir + '/deploy.prototxt',
                    args.caffemodel_dir + 'fcn-8s-cityscapes.caffemodel',
                    caffe.TEST)

    hist_perframe = np.zeros((n_cl, n_cl))
    for idx in range(args.num_test):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, args.num_test))
        # city = idx.split('_')[0]
        # idx is city_shot_frame
        # label = CS.load_label(args.split, city, idx)
        label = CS.load_label(idx)
        im_file = args.result_dir + '/' + idx + '_B_fake_B.jpg'
        im = np.array(Image.open(im_file))
        # im = scipy.misc.imresize(im, (256, 256))
        im = scipy.misc.imresize(im, (label.shape[1], label.shape[2]))
        out = segrun(net, CS.preprocess(im))
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        if args.save_output_images > 0:
            label_im = CS.palette(label)
            pred_im = CS.palette(out)
            scipy.misc.imsave(output_image_dir + '/' + str(idx) + '_pred.jpg', pred_im)
            scipy.misc.imsave(output_image_dir + '/' + str(idx) + '_gt.jpg', label_im)
            scipy.misc.imsave(output_image_dir + '/' + str(idx) + '_input.jpg', im)

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))


main()

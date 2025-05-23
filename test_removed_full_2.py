from model.IFA_matching import IFA_MatchingNet
from util.utils import count_params, set_seed, mIOU

import argparse
import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import glob
from data.dataset import FSSDataset
from util.cosine import cosineSimilarity

def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='fss',
                        choices=['fss', 'deepglobe', 'isic', 'lung'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101', 'resnet50_vdb'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--refine', dest='refine', action='store_true', default=False)
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='batch size of training')

    args = parser.parse_args()
    return args

def predict(model, metric, img_s_list_clone, mask_s_list_clone, img_q, mask_q, cls ):
    img_s_list = img_s_list_clone.permute(1,0,2,3,4)
    mask_s_list = mask_s_list_clone.permute(1,0,2,3)
    
    img_s_list = img_s_list.numpy().tolist()
    mask_s_list = mask_s_list.numpy().tolist()

    img_q, mask_q = img_q.cuda(), mask_q.cuda()

    for k in range(len(img_s_list)):
        img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
        img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

    cls = cls[0].item()
    cls = cls + 1

    with torch.no_grad():
        pred = model(img_s_list, mask_s_list, img_q, None)[0]
        pred = torch.argmax(pred, dim=1)

    pred[pred == 1] = cls
    mask_q[mask_q == 1] = cls

    metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
    return metric 

def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)
    metric_normal = mIOU(num_classes)

    cosine = cosineSimilarity()
    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        similarities = cosine.compute_scores(img_s_list, img_q)
        values = similarities.squeeze()

        img_s_list_new = img_s_list.clone()
        mask_s_list_new = mask_s_list.clone()

        # Get the indices of the two smallest values
        min_indices = torch.topk(values, k=4, largest=False).indices

        # Create a mask with all True values
        mask = torch.ones(img_s_list.size(1), dtype=bool)
        mask[min_indices] = False  # Set the two smallest indices to False

        # Apply the mask to filter out the last two elements
        img_s_filtered = img_s_list_new[:, mask, :, :, :]
        mask_s_filtered = mask_s_list_new[:, mask, :, :]

        # metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
        metric = predict(model, metric, img_s_filtered.clone(), mask_s_filtered.clone(), img_q.clone(), mask_q.clone(), cls )
        metric_normal = predict(model, metric_normal, img_s_list.clone(), mask_s_list.clone(), img_q.clone(), mask_q.clone(), cls )


    return metric.evaluate() * 100.0, metric_normal.evaluate()*100


def main():
    args = parse_args()
    print('\n' + str(args))

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    model = IFA_MatchingNet(args.backbone, args.refine)

    ### Please modify the following paths with your model path if needed.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50' or args.backbone == 'resnet50_vdb':
            if args.shot == 1:
                checkpoint_path = './trained_models/deepglobe/resnet50_1shot_avg_50.63.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/deepglobe/resnet50_5shot_avg_58.76.pth'
    if args.dataset == 'isic' :
        if args.backbone == 'resnet50' or args.backbone == 'resnet50_vdb':
            if args.shot == 1:
                checkpoint_path = './trained_models/isic/resnet50_1shot_avg_66.34.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/isic/resnet50_5shot_avg_69.77.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50' or args.backbone == 'resnet50_vdb':
            if args.shot == 1:
                checkpoint_path = './trained_models/lung/resnet50_1shot_avg_73.96.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/lung/resnet50_5shot_avg_74.59.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50' or args.backbone == 'resnet50_vdb':
            if args.shot == 1:
                checkpoint_path = './trained_models/fss/resnet50_1shot_avg_80.08.pth'
                # checkpoint_path = './trained_models/fss/resnet50_1shot_avg_80.20.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/fss/resnet50_5shot_avg_82.36.pth'


    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    print('\nParams: %.1fM' % count_params(model))

    best_model = DataParallel(model).cuda()

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    total_miou_normal = 0.0 
    model.eval()
    best_model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou, miou_normal = evaluate(best_model, testloader, args)
        total_miou += miou
        total_miou_normal += miou_normal
        
        # miou = evaluate(best_model, testloader, args)
        # total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds custom : %.2f' % (total_miou / 5))
    print('Averaged mIOU on 5 seeds normal : %.2f' % (total_miou_normal / 5))
    print('*' * 32 + '\n')


if __name__ == '__main__':
    main()


from model.IFA_matching import IFA_MatchingNet
from util.utils import count_params, set_seed, mIOU

import argparse
import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import glob
from data.dataset import FSSDataset
import csv 
import numpy as np

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
                        choices=['resnet50', 'resnet101'],
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

filename = "result_one_five_shot.csv"
def evaluate(model_one_shot, model_five_shot, dataloader, args):
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
    # field names
    
    fields = ['D' + str(i) for i in range(args.shot)]
    fields += ['DB', 'Diff', 'Mean']
    # name of csv file
    

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        # print(img_s_list.shape, img_q.shape, 'check size')

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()
        # print(len(img_s_list), len(mask_s_list), 'check 2')
        img_q, mask_q = img_q.cuda(), mask_q.cuda()
        mask_q_c=mask_q.clone().detach()
        mask_q_c=mask_q_c.cuda()

        cls = cls[0].item()
        cls = cls + 1
        # print(cls, 'cls 2')
        one_shot_results = []
        data_result = {}

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()
            # print(img_s_list[k].shape, mask_s_list[k].shape, 'check 3')
            # cls = cls[0].item()
            # cls = cls + 1
            # print(cls, 'cls')

            with torch.no_grad():
                img_s_one_shot = img_s_list[k].unsqueeze(0)
                mask_s_one_shot = mask_s_list[k].unsqueeze(0)
                # print(img_s_one_shot.shape, mask_s_one_shot.shape, 'one shot ccheck')
                
                pred_one = model_one_shot(img_s_one_shot, mask_s_one_shot, img_q, None)[0]
                pred_one = torch.argmax(pred_one, dim=1)
                # print(pred, 'val one shot')
                pred_one[pred_one == 1] = cls
                mask_q_c[mask_q_c == 1] = cls
                
                metric.add_batch(pred_one.cpu().numpy(), mask_q_c.cpu().numpy())
                tbar.set_description("Testing mIOU 1 shot: %.2f" % (metric.evaluate() * 100.0))
                # print("Testing mIOU one shot: %.2f" % (metric.evaluate() * 100.0))
                data_result.update({'D'+ str(k): metric.evaluate() * 100.0})
                one_shot_results.append(metric.evaluate() * 100.0)

        with torch.no_grad():
            pred = model_five_shot(img_s_list, mask_s_list, img_q, None)[0]
            pred = torch.argmax(pred, dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
        tbar.set_description("Testing mIOU full shot: %.2f" % (metric.evaluate() * 100.0))
        result_batch_five_shot = metric.evaluate() * 100.0

        data_result.update({'DB': result_batch_five_shot })
        mean = np.array(one_shot_results).mean()
        data_result.update({'Mean': mean })
        data_result.update({'Diff': mean - result_batch_five_shot })

        # print(data_result)
        with open(filename, "a") as csv_file:
            writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fields)
            writer.writerow(data_result)
            
    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    model_one_shot = IFA_MatchingNet(args.backbone, args.refine)
    model_five_shot = IFA_MatchingNet(args.backbone, args.refine)

    ### Please modify the following paths with your model path if needed.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './trained_models/deepglobe/resnet50_1shot_avg_50.63.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/deepglobe/resnet50_5shot_avg_58.76.pth'
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './trained_models/isic/resnet50_1shot_avg_66.34.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/isic/resnet50_5shot_avg_69.77.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './trained_models/lung/resnet50_1shot_avg_73.96.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/lung/resnet50_5shot_avg_74.59.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                # checkpoint_path = './trained_models/fss/resnet50_1shot_avg_80.08.pth'
                checkpoint_path = './trained_models/fss/resnet50_1shot_avg_80.20.pth'
            if args.shot == 5:
                checkpoint_path = './trained_models/fss/resnet50_5shot_avg_82.36.pth'
    
    checkpoint_one_shot_path = './trained_models/fss/resnet50_1shot_avg_80.08.pth'

    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model_five_shot.load_state_dict(checkpoint)

    checkpoint_one_shot = torch.load(checkpoint_one_shot_path)
    model_one_shot.load_state_dict(checkpoint_one_shot)

    print('\nParams: %.1fM' % count_params(model_one_shot))

    best_one_shot_model = DataParallel(model_one_shot).cuda()
    best_five_shot_model = DataParallel(model_five_shot).cuda()

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model_one_shot.eval()
    model_five_shot.eval()

    fields = ['D' + str(i) for i in range(args.shot)]
    fields += ['DB', 'Diff', 'Mean']
    with open(filename, "a") as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fields)
        writer.writeheader()

    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_one_shot_model, best_five_shot_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')


if __name__ == '__main__':
    main()


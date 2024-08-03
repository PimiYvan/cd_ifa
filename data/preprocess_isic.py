import csv
import os
from shutil import copy

ImgPath = './dataset/ISIC-TMP/ISIC2018_Task1-2_Training_Input'
AnnPath = './dataset/ISIC-TMP/ISIC2018_Task1_Training_GroundTruth'

ImgPathTarget = './dataset/ISIC/ISIC2018_Task1-2_Training_Input'
AnnPathTarget = './dataset/ISIC/ISIC2018_Task1_Training_GroundTruth'


# with open('./isic/class_id.csv', encoding='utf-8-sig') as f:
#     for row in csv.reader(f, skipinitialspace=True):
#         if row[0] != 'ID':
#             imgpath = os.path.join(ImgPath, row[0]) + '.jpg'
#             annpath = os.path.join(AnnPath, row[0]) + '_segmentation.png'
#             if row[1] == 'nevus':
#                 copy(imgpath, os.path.join(ImgPath, '1', os.path.basename(imgpath)))
#                 copy(imgpath, os.path.join(AnnPath, '1', os.path.basename(annpath)))
#             elif row[1] == 'seborrheic_keratosis':
#                 copy(imgpath, os.path.join(ImgPath, '2', os.path.basename(imgpath)))
#                 copy(imgpath, os.path.join(AnnPath, '2', os.path.basename(annpath)))
#             if row[1] == 'melanoma':
#                 copy(imgpath, os.path.join(ImgPath, '3', os.path.basename(imgpath)))
#                 copy(imgpath, os.path.join(AnnPath, '3', os.path.basename(annpath)))


for i in range(1, 4):
    print(i)
    img_folder = os.path.join(ImgPath, str(i))
    ann_folder = os.path.join(AnnPath, str(i))

    img_target_folder = os.path.join(ImgPathTarget, str(i))
    ann_target_folder = os.path.join(AnnPathTarget, str(i))

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(ann_folder, exist_ok=True)

    os.makedirs(img_target_folder, exist_ok=True)
    os.makedirs(ann_target_folder, exist_ok=True)

with open('./data/isic/class_id.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        if row[0] != 'ID':
            imgpath = os.path.join(ImgPath, row[0]) + '.jpg'
            annpath = os.path.join(AnnPath, row[0]) + '_segmentation.png'

            if row[1] == 'nevus':
                copy(imgpath, os.path.join(ImgPathTarget, '1', os.path.basename(imgpath)))
                copy(imgpath, os.path.join(AnnPathTarget, '1', os.path.basename(annpath)))
            elif row[1] == 'seborrheic_keratosis':
                copy(imgpath, os.path.join(ImgPathTarget, '2', os.path.basename(imgpath)))
                copy(imgpath, os.path.join(AnnPathTarget, '2', os.path.basename(annpath)))
            if row[1] == 'melanoma':
                copy(imgpath, os.path.join(ImgPathTarget, '3', os.path.basename(imgpath)))
                copy(imgpath, os.path.join(AnnPathTarget, '3', os.path.basename(annpath)))

    
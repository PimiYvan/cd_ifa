

git clone https://github.com/niejiahao1998/IFA.git <br/>
git clone https://github.com/PimiYvan/cd_ifa.git <br/>

chmod +x dataset.sh 
cd ifa

python3.10 -m venv env <br/> [mii] loading StdEnv/2023 python/3.10.13
source env/bin/activate <br/>

cd cd_ifa
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
mv ./VOCdevkit ./dataset
mv SegmentationClassAug ./dataset/VOC2012/
mv ./dataset/VOCdevkit/VOC2012 ./dataset/
rm ./dataset/VOCdevkit -r

gdown 10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2
unzip SegmentationClassAug.zip

gdown 16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI
unzip fewshot_data.zip
mv ./fewshot_data/* ./dataset
rm -r ./fewshot_data
mv ./dataset/fewshot_data ./dataset/FSS-1000


pip install --upgrade pip <br/>
pip install -r requirements.txt <br/>

mv ./dataset/VOCdevkit/VOC2012 ./dataset/
rm ./dataset/VOCdevkit -r

mkdir ./dataset/Deepglobe ./dataset/Deepglobe/01_train_ori
mkdir ./dataset/Deepglobe/02_train_crop
mkdir ./dataset/Deepglobe/03_train_filter
mkdir ./dataset/Deepglobe/04_train_cat

gdown 1kUEFWv-DByaH5zuBBgN_v1XnI3Jn3GYi
unzip deeplobe_ifa.zip
rm deeplobe_ifa.zip

salloc --time=1:0:0 --mem=3G --ntasks=2 --account=def-menna --gres=gpu:1 --nodes=1
salloc --time=1:0:0 --mem=34G --ntasks=2 --account=rrg-menna --gres=gpu:1 --nodes=1

pip install -U git+https://github.com/albu/albumentations


mkdir pretrained
gdown 1SgzQSBUp29dDWpq_rdPK114leBo5VXt-
mv ./resnet50.pth ./pretrained/
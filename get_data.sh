mkdir coco
cd coco
mkdir images
cd images
wget http://images.cocodataset.org/zips/train2017.zip
unzip -qq train2017.zip
rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip -qq val2017.zip
rm val2017.zip
cd ..
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
unzip -qq annotations_trainval2017.zip
unzip -qq stuff_annotations_trainval2017.zip
rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
cd ..
wget https://github.com/chechaohp/test_repo/releases/download/pretrain/pose_higher_hrnet_w32_512_2.pth
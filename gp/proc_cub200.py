
## Organizing data

import os
import pandas as pd
import shutil
import cv2


## Arguments 

dir_cub200 = './data/CUB_200_2011'
crop_by_bbox = True


## Load metadata

dir_cub200_images = os.path.join(dir_cub200, 'images')

paths_df = pd.read_csv(os.path.join(dir_cub200, 'images.txt'),
                      header=None, delimiter=' ', names=['id', 'rel_path'])
split_df = pd.read_csv(os.path.join(dir_cub200, 'train_test_split.txt'),
            header=None, delimiter=' ', names=['id', 'split'])
bbox_df = pd.read_csv(os.path.join(dir_cub200, 'bounding_boxes.txt'),
            header=None, delimiter=' ', names=['id', 'x', 'y', 'width', 'height'])

meta_df = pd.merge(paths_df, split_df, on=['id'], how='inner')
meta_df = pd.merge(meta_df, bbox_df, on=['id'], how='inner')
meta_df['class'] = meta_df['rel_path'].apply(lambda x: x.split('/')[0])
meta_df

## move images to train/test folders

if crop_by_bbox:
    dir_train = os.path.join(dir_cub200, 'cropped_images_train')
    dir_test  = os.path.join(dir_cub200, 'cropped_images_test')
else:
    dir_train = os.path.join(dir_cub200, 'images_train')
    dir_test  = os.path.join(dir_cub200, 'images_test')
    
os.makedirs(dir_train, exist_ok=True)
os.makedirs(dir_test,  exist_ok=True)


for idx, row in meta_df.iterrows():
    class_name, jpg_name = row['rel_path'].split('/')
    dir_train_or_test = dir_train if ( row['split'] == 1 ) else dir_test

    dir_dst = os.path.join(dir_train_or_test, class_name)
    os.makedirs(dir_dst, exist_ok=True)

    im_src = os.path.join(dir_cub200_images, row['rel_path'])
    im_dst = os.path.join(dir_dst, jpg_name)
    
    if crop_by_bbox:
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['width'])
        h = int(row['height'])
        
        im = cv2.imread(im_src)

        ## crop by bbox, as square as possible

        H, W, C = im.shape
        if w>h:
            side_len = w
            xx = x
            yy = int(max(0, y+(h-side_len)/2))
            ww = w
            hh = int(min(H-1, yy+side_len))-yy
        else:
            side_len = h
            xx = int(max(0, x+(w-side_len)/2))
            yy = y
            ww = int(min(W-1, xx+side_len))-xx
            hh = h

        im_crop = im[yy:yy+hh,xx:xx+ww,:]
        cv2.imwrite(im_dst, im_crop)
    else:
        shutil.copy(src=im_src, dst=im_dst)
    
    if idx % 200 == 0:
        print(f'[{idx}\t/{len(meta_df)}]')
    
import cv2
import os
import numpy as np
import random
from tools.tool import *

path="2-GT"
ratio=10
mask_path=os.path.join(path,"human-mask")
new_mask_path=os.path.join(path,str(ratio)+"mask")
createFolder(new_mask_path,clean=True)
mask_list=os.listdir(mask_path)
mask_list.sort()
for name in  mask_list:
    mask=cv2.imread(os.path.join(mask_path,name),-1)
    ret, mask = cv2.connectedComponents(mask, ltype=2)
    num_list=np.unique(mask)[1:]
    length=len(num_list)
    num=int(length*(1-0.01*ratio))
    remove_list=random.sample(list(num_list),num)
    print(num_list)
    print(remove_list)

    for label in remove_list:
        remove=(mask==label)
        mask-=remove*label
    print(np.unique(mask))
    mask=(mask>0)*255

    mask=mask.astype(np.uint8)
    cv2.imwrite(os.path.join(new_mask_path,name),mask)

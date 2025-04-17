import numpy as np
import cv2
import os

def showImgWithMask(img,mask):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    '''
    mask_channel1 = (np.ones((mask.shape)) * 255).astype(np.uint8)
    mask_channel2 = 255 - mask
    mask = np.stack((mask_channel2, mask_channel2, mask_channel1), axis=2)'''


    overlay = cv2.addWeighted(img_rgb, 1.5, mask, 0.5, 0)
    return overlay

drawing = False
def draw_circle(event,x,y,flags,param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = 2
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == 1:
            cv2.rectangle(mask,(x-1,y-1),(x+1,y+1),(0,0,255),thickness=-1)
            print("1")
        if drawing == 2:
            cv2.rectangle(mask,(x-2,y-2),(x+2,y+2),(0,0,0),thickness=-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = 3
        cv2.rectangle(mask,(x-1,y-1),(x+1,y+1),(0,0,255),thickness=-1)
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 4
        cv2.rectangle(mask,(x-2,y-2),(x+2,y+2),(0,0,0),thickness=-1)



img_path="data/imgs"
mask_path="data/mask"
mask_new_path="data/mask_new"
#mask = np.zeros((746,770,3), np.uint8)
i=0
mask=cv2.imread(os.path.join(mask_path,str(i).zfill(6)+".tif"),-1)
mask_channel2=np.zeros((746,770), np.uint8)
mask = np.stack((mask_channel2, mask_channel2, mask), axis=-1)
img=cv2.imread(os.path.join(img_path,str(i).zfill(6)+".tif"),-1)
cv2.namedWindow('image'+str(i),cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('image'+str(i),draw_circle)

while(1):
    cv2.imshow('image'+str(i),showImgWithMask(img,mask))
    if cv2.waitKey(1)==ord('s'):
        cv2.destroyWindow('image'+str(i))
        cv2.namedWindow('mask'+str(i),cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('mask'+str(i), draw_circle)
        while(1):
            cv2.imshow('mask'+str(i),mask)
            if cv2.waitKey(1)==ord('d'):
                cv2.destroyWindow('mask'+str(i))
                cv2.imwrite(os.path.join(mask_new_path,str(i).zfill(6)+".tif"),mask[:,:,-1])
                break
        i+=1
        mask = cv2.imread(os.path.join(mask_path, str(i).zfill(6) + ".tif"), -1)
        mask_channel2 = np.zeros((746, 770), np.uint8)
        mask = np.stack((mask_channel2, mask_channel2, mask), axis=-1)
        img = cv2.imread(os.path.join(img_path, str(i).zfill(6) + ".tif"), -1)
        cv2.namedWindow('image'+str(i), cv2.WINDOW_KEEPRATIO)


cv2.destroyAllWindows()

import cv2
import numpy as np
import os

def binarization(img,threshold):
    img=np.uint8(img)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img

def remove_uneven_illumination(img, blur_kernel_size=501):
    if blur_kernel_size%2==0:
        blur_kernel_size=blur_kernel_size+1
    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)
    img_blur = cv2.GaussianBlur(img_f, (blur_kernel_size, blur_kernel_size), 0)
    result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.uint8)
    return result

def useAreaFilter(img,area_size):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_new = np.stack((img, img, img), axis=2)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area < area_size:
            img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))
    img = img_new[:, :, 0]
    return img

def median_filter(img,median_blur_size):
    if median_blur_size % 2 == 0:
        median_blur_size += 1
    img = cv2.medianBlur(img, median_blur_size)
    return img

def update(x):
    global img, img2
    alpha = cv2.getTrackbarPos('Alpha', 'scale')
    beta = cv2.getTrackbarPos('Beta', 'scale')
    alpha = alpha * 0.01
    blur_kernel_size = cv2.getTrackbarPos('Gauss', 'scale')
    threshold=cv2.getTrackbarPos('Threshold', 'scale')

    area_size=cv2.getTrackbarPos('AF','scale')
    median_blur_size=cv2.getTrackbarPos('Median','scale')

    kernel_size=cv2.getTrackbarPos('E+D1','scale')
    median_blur_size1=cv2.getTrackbarPos('Median1','scale')
    area_size1=cv2.getTrackbarPos('AF1','scale')

    kernel_size1=cv2.getTrackbarPos('D+E1','scale')
    median_blur_size2=cv2.getTrackbarPos('Median2','scale')
    area_size2=cv2.getTrackbarPos('AF2','scale')

    kernel_size2 = cv2.getTrackbarPos('E+D2', 'scale')
    median_blur_size3=cv2.getTrackbarPos('Median3','scale')
    area_size3 = cv2.getTrackbarPos('AF3', 'scale')

    kernel_size3=cv2.getTrackbarPos('D+E2','scale')
    median_blur_size4=cv2.getTrackbarPos('Median4','scale')
    area_size4=cv2.getTrackbarPos('AF4','scale')

    kernel_size4 = cv2.getTrackbarPos('E+D3', 'scale')
    median_blur_size5=cv2.getTrackbarPos('Median5','scale')
    area_size5 = cv2.getTrackbarPos('AF5', 'scale')

    kernel_size5=cv2.getTrackbarPos('D+E3','scale')
    median_blur_size6=cv2.getTrackbarPos('Median6','scale')
    area_size6=cv2.getTrackbarPos('AF6','scale')

    kernel_size6 = cv2.getTrackbarPos('E+D4', 'scale')
    median_blur_size7=cv2.getTrackbarPos('Median7','scale')
    area_size7 = cv2.getTrackbarPos('AF7', 'scale')
    '''
    kernel_size7=cv2.getTrackbarPos('D+E5','scale')
    median_blur_size8=cv2.getTrackbarPos('Median8','scale')
    area_size8=cv2.getTrackbarPos('AF8','scale')
'''
    sample = remove_uneven_illumination(np.uint8(np.clip((alpha * img2 + beta), 0, 255)), blur_kernel_size)
    sample = binarization(sample, threshold).astype(np.uint8)
    if area_size!=0:
        sample=useAreaFilter(sample, area_size)
    if median_blur_size!=0:
        sample=median_filter(sample,median_blur_size)

    if kernel_size!=0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        sample = cv2.morphologyEx(sample, cv2.MORPH_OPEN, kernel)  # 先腐蚀再膨胀
    if median_blur_size1!=0:
        sample=median_filter(sample,median_blur_size1)
    if area_size1!=0:
        sample=useAreaFilter(sample, area_size1)

    if kernel_size1!=0:
        kernel1= cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size1,kernel_size1))
        sample=cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel1)
    if median_blur_size2!=0:
        sample=median_filter(sample,median_blur_size2)
    if area_size2!=0:
        sample=useAreaFilter(sample, area_size2)

    if kernel_size2!=0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size2,kernel_size2))#生成3*3大小的矩形
        sample = cv2.morphologyEx(sample, cv2.MORPH_OPEN, kernel2)
    if median_blur_size3!=0:
        sample=median_filter(sample,median_blur_size3)
    if area_size3!=0:
        sample=useAreaFilter(sample, area_size3)

    if kernel_size3!=0:
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size3, kernel_size3))
        sample = cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel3)
    if median_blur_size4!=0:
        sample=median_filter(sample,median_blur_size4)
    if area_size4!=0:
        sample=useAreaFilter(sample, area_size4)

    if kernel_size4 != 0:
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size4, kernel_size4))
        sample = cv2.morphologyEx(sample, cv2.MORPH_OPEN, kernel4)
    if median_blur_size5 != 0:
        sample = median_filter(sample, median_blur_size5)
    if area_size5 != 0:
        sample = useAreaFilter(sample, area_size5)

    if kernel_size5 != 0:
        kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size5, kernel_size5))
        sample = cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel5)
    if median_blur_size6 != 0:
        sample = median_filter(sample, median_blur_size6)
    if area_size6 != 0:
        sample = useAreaFilter(sample, area_size6)

    if kernel_size6 != 0:
        kernel6 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size6, kernel_size6))
        sample = cv2.morphologyEx(sample, cv2.MORPH_OPEN, kernel6)
    if median_blur_size7 != 0:
        sample = median_filter(sample, median_blur_size7)
    if area_size7 != 0:
        sample = useAreaFilter(sample, area_size7)
    '''
    if kernel_size7!=0:
        kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size7, kernel_size7))
        sample = cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel7)
    if median_blur_size8 != 0:
        sample = median_filter(sample, median_blur_size8)
    if area_size8 != 0:
        sample = useAreaFilter(sample, area_size8)
'''
    img=sample





alpha = 10
beta = 100
blur_kernel_size=701
threshold=60
img=img2=cv2.imread("red_flore/0.tif",-1)
image=cv2.imread("brightfield/0.tif",-1)
cv2.namedWindow('scale',cv2.WINDOW_NORMAL)
cv2.createTrackbar('Alpha', 'scale', 0, 300, update)
cv2.createTrackbar('Beta', 'scale', 0, 255, update)
cv2.createTrackbar('Gauss', 'scale', 1, 1001, update)
cv2.createTrackbar('Threshold', 'scale', 0, 255, update)
cv2.createTrackbar('AF','scale',0,300,update)
cv2.createTrackbar('Median','scale',0,20,update)

cv2.createTrackbar('E+D1', 'scale', 0, 50, update)
cv2.createTrackbar('Median1','scale',0,20,update)
cv2.createTrackbar('AF1','scale',0,300,update)

cv2.createTrackbar('D+E1', 'scale', 0, 50, update)
cv2.createTrackbar('Median2','scale',0,20,update)
cv2.createTrackbar('AF2','scale',0,300,update)

cv2.createTrackbar('E+D2', 'scale', 0, 50, update)
cv2.createTrackbar('Median3','scale',0,20,update)
cv2.createTrackbar('AF3','scale',0,300,update)

cv2.createTrackbar('D+E2', 'scale', 0, 50, update)
cv2.createTrackbar('Median4','scale',0,20,update)
cv2.createTrackbar('AF4','scale',0,300,update)

cv2.createTrackbar('E+D3', 'scale', 0, 50, update)
cv2.createTrackbar('Median5','scale',0,20,update)
cv2.createTrackbar('AF5','scale',0,300,update)

cv2.createTrackbar('D+E3', 'scale', 0, 50, update)
cv2.createTrackbar('Median6','scale',0,20,update)
cv2.createTrackbar('AF6','scale',0,300,update)

cv2.createTrackbar('E+D4', 'scale', 0, 50, update)
cv2.createTrackbar('Median7','scale',0,20,update)
cv2.createTrackbar('AF7','scale',0,300,update)
'''

cv2.createTrackbar('D+E4', 'scale', 0, 50, update)
cv2.createTrackbar('Median8','scale',0,20,update)
cv2.createTrackbar('AF8','scale',0,300,update)
'''

cv2.setTrackbarPos('Alpha', 'scale', 10)
cv2.setTrackbarPos('Beta', 'scale', 100)
cv2.setTrackbarPos('Gauss', 'scale', 701)
cv2.setTrackbarPos('Threshold', 'scale', 0)
cv2.setTrackbarPos('AF','scale',0)
cv2.setTrackbarPos('Median','scale',0)

cv2.setTrackbarPos('E+D1', 'scale', 0)
cv2.setTrackbarPos('Median1','scale',0)
cv2.setTrackbarPos('AF1','scale',0)

cv2.setTrackbarPos('D+E1', 'scale', 0)
cv2.setTrackbarPos('Median2','scale',0)
cv2.setTrackbarPos('AF2','scale',0)

cv2.setTrackbarPos('E+D2', 'scale', 0)
cv2.setTrackbarPos('Median3','scale',0)
cv2.setTrackbarPos('AF3','scale',0)

cv2.setTrackbarPos('D+E2', 'scale', 0)
cv2.setTrackbarPos('Median4','scale',0)
cv2.setTrackbarPos('AF4','scale',0)

cv2.setTrackbarPos('E+D3', 'scale', 0)
cv2.setTrackbarPos('Median5','scale',0)
cv2.setTrackbarPos('AF5','scale',0)

cv2.setTrackbarPos('D+E3', 'scale', 0)
cv2.setTrackbarPos('Median6','scale',0)
cv2.setTrackbarPos('AF6','scale',0)

cv2.setTrackbarPos('E+D4', 'scale', 0)
cv2.setTrackbarPos('Median7','scale',0)
cv2.setTrackbarPos('AF7','scale',0)
'''
cv2.setTrackbarPos('D+E4', 'scale', 0)
cv2.setTrackbarPos('Median8','scale',0)
cv2.setTrackbarPos('AF8','scale',0)
'''

img_root="brightfield/"
mask_root="red_flore/"
n=len(os.listdir(img_root))
i=0
mask_path="red_flore/0.tif"
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
while(1):

    cv2.imshow('image',image)
    cv2.imshow('mask', img)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite(mask_path,img)
        print(mask_path, "has been processed!")
        i+=1
        img_path = os.path.join(img_root, "%d.tif" % (i))
        mask_path = os.path.join(mask_root, "%d.tif" % (i))
        img=img2=cv2.imread(mask_path,-1)
        image=cv2.imread(img_path,-1)


        alpha = cv2.getTrackbarPos('Alpha', 'scale')
        beta = cv2.getTrackbarPos('Beta', 'scale')
        blur_kernel_size = cv2.getTrackbarPos('Gauss', 'scale')
        threshold = cv2.getTrackbarPos('Threshold', 'scale')

        area_size = cv2.getTrackbarPos('AF', 'scale')
        median_blur_size = cv2.getTrackbarPos('Median', 'scale')

        kernel_size = cv2.getTrackbarPos('E+D1', 'scale')
        median_blur_size1 = cv2.getTrackbarPos('Median1', 'scale')
        area_size1 = cv2.getTrackbarPos('AF1', 'scale')

        kernel_size1 = cv2.getTrackbarPos('D+E1', 'scale')
        median_blur_size2 = cv2.getTrackbarPos('Median2', 'scale')
        area_size2 = cv2.getTrackbarPos('AF2', 'scale')

        kernel_size2 = cv2.getTrackbarPos('E+D2', 'scale')
        median_blur_size3 = cv2.getTrackbarPos('Median3', 'scale')
        area_size3 = cv2.getTrackbarPos('AF3', 'scale')

        kernel_size3 = cv2.getTrackbarPos('D+E2', 'scale')
        median_blur_size4 = cv2.getTrackbarPos('Median4', 'scale')
        area_size4 = cv2.getTrackbarPos('AF4', 'scale')

        kernel_size4 = cv2.getTrackbarPos('E+D3', 'scale')
        median_blur_size5 = cv2.getTrackbarPos('Median5', 'scale')
        area_size5 = cv2.getTrackbarPos('AF5', 'scale')

        kernel_size5 = cv2.getTrackbarPos('D+E3', 'scale')
        median_blur_size6 = cv2.getTrackbarPos('Median6', 'scale')
        area_size6 = cv2.getTrackbarPos('AF6', 'scale')

        kernel_size6 = cv2.getTrackbarPos('E+D4', 'scale')
        median_blur_size7 = cv2.getTrackbarPos('Median7', 'scale')
        area_size7 = cv2.getTrackbarPos('AF7', 'scale')

        cv2.setTrackbarPos('Alpha', 'scale', 10)
        cv2.setTrackbarPos('Beta', 'scale', 100)
        cv2.setTrackbarPos('Gauss', 'scale', 701)
        cv2.setTrackbarPos('Threshold', 'scale', 0)
        cv2.setTrackbarPos('AF', 'scale', 0)
        cv2.setTrackbarPos('Median', 'scale', 0)

        cv2.setTrackbarPos('E+D1', 'scale', 0)
        cv2.setTrackbarPos('Median1', 'scale', 0)
        cv2.setTrackbarPos('AF1', 'scale', 0)

        cv2.setTrackbarPos('D+E1', 'scale', 0)
        cv2.setTrackbarPos('Median2', 'scale', 0)
        cv2.setTrackbarPos('AF2', 'scale', 0)

        cv2.setTrackbarPos('E+D2', 'scale', 0)
        cv2.setTrackbarPos('Median3', 'scale', 0)
        cv2.setTrackbarPos('AF3', 'scale', 0)

        cv2.setTrackbarPos('D+E2', 'scale', 0)
        cv2.setTrackbarPos('Median4', 'scale', 0)
        cv2.setTrackbarPos('AF4', 'scale', 0)

        cv2.setTrackbarPos('E+D3', 'scale', 0)
        cv2.setTrackbarPos('Median5', 'scale', 0)
        cv2.setTrackbarPos('AF5', 'scale', 0)

        cv2.setTrackbarPos('D+E3', 'scale', 0)
        cv2.setTrackbarPos('Median6', 'scale', 0)
        cv2.setTrackbarPos('AF6', 'scale', 0)

        cv2.setTrackbarPos('E+D4', 'scale', 0)
        cv2.setTrackbarPos('Median7', 'scale', 0)
        cv2.setTrackbarPos('AF7', 'scale', 0)
 
        cv2.setTrackbarPos('Alpha', 'scale', alpha)
        cv2.setTrackbarPos('Beta', 'scale', beta)
        cv2.setTrackbarPos('Gauss', 'scale', blur_kernel_size)
        cv2.setTrackbarPos('Threshold', 'scale', threshold)
        cv2.setTrackbarPos('AF', 'scale', area_size)
        cv2.setTrackbarPos('Median', 'scale', median_blur_size)

        cv2.setTrackbarPos('E+D1', 'scale', kernel_size)
        cv2.setTrackbarPos('Median1', 'scale', median_blur_size1)
        cv2.setTrackbarPos('AF1', 'scale', area_size1)

        cv2.setTrackbarPos('D+E1', 'scale', kernel_size1)
        cv2.setTrackbarPos('Median2', 'scale', median_blur_size2)
        cv2.setTrackbarPos('AF2', 'scale', area_size2)

        cv2.setTrackbarPos('E+D2', 'scale', kernel_size2)
        cv2.setTrackbarPos('Median3', 'scale', median_blur_size3)
        cv2.setTrackbarPos('AF3', 'scale', area_size3)

        cv2.setTrackbarPos('D+E2', 'scale', kernel_size3)
        cv2.setTrackbarPos('Median4', 'scale', median_blur_size4)
        cv2.setTrackbarPos('AF4', 'scale', area_size4)

        cv2.setTrackbarPos('E+D3', 'scale', kernel_size2)
        cv2.setTrackbarPos('Median5', 'scale', median_blur_size5)
        cv2.setTrackbarPos('AF5', 'scale', area_size5)

        cv2.setTrackbarPos('D+E3', 'scale', kernel_size3)
        cv2.setTrackbarPos('Median6', 'scale', median_blur_size6)
        cv2.setTrackbarPos('AF6', 'scale', area_size6)

        cv2.setTrackbarPos('E+D4', 'scale', kernel_size2)
        cv2.setTrackbarPos('Median7', 'scale', median_blur_size7)
        cv2.setTrackbarPos('AF7', 'scale', area_size7)


        '''
        cv2.setTrackbarPos('Alpha', 'scale', 10)
        cv2.setTrackbarPos('Beta', 'scale', 100)
        cv2.setTrackbarPos('Gauss', 'scale', 701)
        cv2.setTrackbarPos('Threshold', 'scale', 0)
        cv2.setTrackbarPos('AF', 'scale', 0)
        cv2.setTrackbarPos('Median', 'scale', 0)

        cv2.setTrackbarPos('E+D1', 'scale', 0)
        cv2.setTrackbarPos('Median1', 'scale', 0)
        cv2.setTrackbarPos('AF1', 'scale', 0)

        cv2.setTrackbarPos('D+E1', 'scale', 0)
        cv2.setTrackbarPos('Median2', 'scale', 0)
        cv2.setTrackbarPos('AF2', 'scale', 0)

        cv2.setTrackbarPos('E+D2', 'scale', 0)
        cv2.setTrackbarPos('Median3', 'scale', 0)
        cv2.setTrackbarPos('AF3', 'scale', 0)

        cv2.setTrackbarPos('D+E2', 'scale', 0)
        cv2.setTrackbarPos('Median4', 'scale', 0)
        cv2.setTrackbarPos('AF4', 'scale', 0)

        cv2.setTrackbarPos('E+D3', 'scale', 0)
        cv2.setTrackbarPos('Median5', 'scale', 0)
        cv2.setTrackbarPos('AF5', 'scale', 0)

        cv2.setTrackbarPos('D+E3', 'scale', 0)
        cv2.setTrackbarPos('Median6', 'scale', 0)
        cv2.setTrackbarPos('AF6', 'scale', 0)

        cv2.setTrackbarPos('E+D4', 'scale', 0)
        cv2.setTrackbarPos('Median7', 'scale', 0)
        cv2.setTrackbarPos('AF7', 'scale', 0)
        '''
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()

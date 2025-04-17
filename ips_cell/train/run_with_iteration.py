import os.path
import subprocess
from track.track_3_period import *
from detect.test import *
from detect.train import *
from detect.net import Unet,NestedUNet,FCN8s,AttU_Net
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
import xlwt
from track.track_3_period import track_all_periods
from temporal_analysis import reduceFP_with_Pool

def augBC(imgs_dir,imgs_aug_dir,sequence,mask_dir,mask_aug_dir,mask):
    contrast_list = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
    contrast_type=len(contrast_list)+1
    img_num=sequence
    img_name=str(sequence).zfill(6)+".tif"
    img=cv2.imread(os.path.join(imgs_dir,img_name),-1)
    cv2.imwrite(os.path.join(imgs_aug_dir,str(img_num*contrast_type).zfill(6)+".tif"),img)
    if mask:
        img_mask=cv2.imread(os.path.join(mask_dir,img_name),-1)
        for i in range(contrast_type):
            cv2.imwrite(os.path.join(mask_aug_dir, str(img_num * contrast_type+i).zfill(6) + ".tif"), img_mask)
    img=img.astype(np.float32)
    print("generating contrast augmentation for image:",img_name)
    for j,contrast in enumerate(contrast_list,start=1):
        img_C=np.clip(img*contrast,0,255)
        img_C=img_C.astype(np.uint8)
        cv2.imwrite(os.path.join(imgs_aug_dir,str(img_num*contrast_type+j).zfill(6)+".tif"),img_C)

def augmentationWithPool(imgs_dir,imgs_aug_dir,mask_dir=None,mask_aug_dir=None,mask=False):
    createFolder(imgs_aug_dir,clean=True)
    if mask:
        createFolder(mask_aug_dir,clean=True)
    img_list=os.listdir(imgs_dir)
    img_list.sort()
    print(img_list)
    p=multiprocessing.Pool()
    for i in range(len(img_list)):
        p.apply_async(augBC,args=(imgs_dir,imgs_aug_dir,i,mask_dir,mask_aug_dir,mask,))
    p.close()
    p.join()

'''def test_and_process(img_path,result_path,process_result_path,wighted_sum_path,mask_new_path,ckpt_path,mask_old_path):
    createFolder(result_path,clean=True)
    createFolder(process_result_path,clean=True)
    createFolder(mask_new_path,clean=True)
    createFolder(wighted_sum_path, clean=True)
    ckpt_list=os.listdir(ckpt_path)
    ckpt_list.sort()
    final_ckpt=os.path.join(ckpt_path,ckpt_list[-1])
    print("predicting images with ckpt:",final_ckpt)
    test(img_path,result_path,final_ckpt)
    result_list=os.listdir(result_path)
    result_list=[name for name in result_list if ".tif" in name]
    result_list.sort()
    result_imgs=[]
    print("\tprocessing predicted mask:",result_path)
    for result_name in result_list:
        result=cv2.imread(os.path.join(result_path,result_name),-1)
        img=result.astype(np.float32)
        result_imgs.append(img)
        ret,result=cv2.threshold(result,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(process_result_path,result_name),result)
    print("\tgenerating new mask:",mask_new_path)

    contrast_type=int(len(result_imgs)/233)
    for i in range(0,len(result_imgs),contrast_type):
        mask_combine=np.zeros((result_imgs[i].shape),dtype=np.float32)
        for j in range(i,i+contrast_type):
            mask_combine+=result_imgs[j]*0.125
        mask_new=mask_combine.astype(np.uint8)
        cv2.imwrite(os.path.join(wighted_sum_path, str(i//contrast_type).zfill(6) + ".tif"), mask_new)
        #ret, mask_new = cv2.threshold(mask_new, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #mask_new = useAreaFilter(mask_new, 40)

        #加上初始标签
        mask_old=cv2.imread(os.path.join(mask_old_path,str(i).zfill(6)+".tif"),-1)
        if mask_old.shape[0] != 736:
            mask_old = mask_old[5:741, 1:769]
            mask_old = mask_old.astype(np.float32)
            mask_final = 0.5 * mask_old + 0.5 * mask_combine
            mask_final = mask_final.astype(np.uint8)
            ret, mask_final = cv2.threshold(mask_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_final = useAreaFilter(mask_final, 40)
            cv2.imwrite(os.path.join(mask_new_path, str(i + k).zfill(6) + ".tif"), mask_final)
        #加上初始标签引导网络走向
        for k in range(contrast_type):
            mask_old=cv2.imread(os.path.join(mask_old_path,str(i+k).zfill(6)+".tif"),-1)
            if mask_old.shape[0]!=736:
                mask_old=mask_old[5:741,1:769]
            mask_old=mask_old.astype(np.float32)
            mask_final=0.5*mask_old+0.5*mask_combine
            mask_final=mask_final.astype(np.uint8)
            ret, mask_final = cv2.threshold(mask_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_final = useAreaFilter(mask_final, 40)
            cv2.imwrite(os.path.join(mask_new_path,str(i+k).zfill(6)+".tif"),mask_final)'''

def test_and_process(img_path,img_aug_path,ckpt_path,current_path,next_path):
    predict_path=os.path.join(current_path,"predict")
    predict_result_path=os.path.join(os.path.join(current_path,"predict_result"))
    weighted_sum_path=os.path.join(current_path,"weighted_sum")
    temporal_path=os.path.join(current_path,"temporal")
    temporal_mask_path=os.path.join(temporal_path,"mask")
    next_mask_path=os.path.join(next_path,"mask")
    mask_path=os.path.join(current_path,"mask")
    createFolder(predict_path, clean=True)
    createFolder(predict_result_path, clean=True)
    createFolder(weighted_sum_path, clean=True)
    createFolder(temporal_path, clean=True)
    createFolder(next_mask_path, clean=True)
    createFolder(temporal_mask_path,clean=True)
    ckpt_list = os.listdir(ckpt_path)
    ckpt_list.sort()
    final_ckpt = os.path.join(ckpt_path, ckpt_list[-1])

    #predict
    print("predicting images with ckpt:", final_ckpt)
    test(img_aug_path, predict_path, final_ckpt)
    predict_list = os.listdir(predict_path)
    predict_list = [name for name in predict_list if ".tif" in name]
    predict_list.sort()
    predict_imgs = []
    print("\tprocessing predicted mask in:", predict_path)
    for name in predict_list:
        result = cv2.imread(os.path.join(predict_path, name), -1)
        img = result.astype(np.float32)
        predict_imgs.append(img)
        ret, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(predict_result_path, name), result)

    #加权和并结合初始标签
    contrast_type = int(len(predict_imgs) / 233)
    for i in range(0, len(predict_imgs), contrast_type):
        mask_combine = np.zeros((predict_imgs[i].shape), dtype=np.float32)
        for j in range(i, i + contrast_type):
            mask_combine += predict_imgs[j] * 0.125
        weighted_sum = mask_combine.astype(np.uint8)
        cv2.imwrite(os.path.join(weighted_sum_path, str(i // contrast_type).zfill(6) + ".tif"), weighted_sum)
        '''ret, mask_new = cv2.threshold(mask_new, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_new = useAreaFilter(mask_new, 40)'''
        # 加上初始标签
        mask_old = cv2.imread(os.path.join(mask_path, str(i).zfill(6) + ".tif"), -1)
        ret,mask_combine_binary=cv2.threshold(weighted_sum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask_old.shape[0] != 736:
            mask_old = mask_old[5:741, 1:769]
        mask_final = mask_old + mask_combine_binary
        mask_final = np.clip(mask_final,0,255)
        mask_final=mask_final.astype(np.uint8)
        mask_final = useAreaFilter(mask_final, 40)
        cv2.imwrite(os.path.join(temporal_mask_path, str(i // contrast_type).zfill(6) + ".tif"), mask_final)

    #追踪分析
    print("\tstart analyzing by temporal consistency")
    temporal_track_path=os.path.join(temporal_path,"track")
    track_all_periods(img_aug_path,temporal_mask_path,temporal_track_path,remove_edge=False)
    reduceFP_with_Pool(img_path,temporal_track_path)
    period_list=os.listdir(temporal_track_path)
    period_list.sort()
    print("\tgenerating new mask")
    for period_name in period_list:
        period_path=os.path.join(temporal_track_path,period_name)
        remove_FP_path=os.path.join(period_path,"remove_FP")
        length=len(os.listdir(next_mask_path))
        mask_without_FP_list=os.listdir(remove_FP_path)
        mask_without_FP_list.sort()
        for j in range(len(mask_without_FP_list)):
            mask=cv2.imread(os.path.join(remove_FP_path,mask_without_FP_list[j]),-1)
            for k in range(contrast_type):
                cv2.imwrite(os.path.join(next_mask_path,str(length+j*contrast_type+k).zfill(6)+".tif"),mask)


def trainWithIteration(img_path,img_aug_path,iteration_path,keep_training,iteration_times=5,batch_size=32,data_shuffle=False):
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    #model = Unet(1, 1)
    #model=NestedUNet(1,1)
    #model=FCN8s(1,1)
    model=AttU_Net(1,1)
    criterion = nn.BCEWithLogitsLoss()  # 包含sigmoid
    optimizer = optim.Adam(model.parameters())
    ckpt_path = os.path.join(iteration_path, "checkpoints")
    createFolder(ckpt_path,clean=True)

    for i in range(iteration_times):
        if i==0:
            training=False
        else:
            training=keep_training
        current_folder=os.path.join(iteration_path,"iteration"+str(i+1).zfill(2))
        mask_path=os.path.join(current_folder,"mask")
        print("\nstart iteration {}/{}: {}----{}".format(i + 1,iteration_times, img_aug_path,mask_path))
        data = TrainDataset(img_aug_path, mask_path, x_transforms, y_transforms)
        dataloader = DataLoader(data, batch_size, shuffle=data_shuffle, num_workers=4)
        train_model(model, criterion, optimizer, dataloader, training, ckpt_path, num_epochs=1)

        next_folder=os.path.join(iteration_path,"iteration"+str(i+2).zfill(2))
        createFolder(next_folder)
        #test_and_process(img_path,result_path,process_result_path,wighted_sum_path,mask_new_path,ckpt_path,mask_path)
        test_and_process(img_path,img_aug_path,ckpt_path,current_folder,next_folder)

def evaluate_DET_TRA(bright_field,period,track_result_path,isWin=False):
    period_num=str(period).zfill(2)
    res_path=os.path.join(track_result_path,period_num+"/"+period_num+"_RES")
    bright_field_num=str(bright_field).zfill(2)
    data_folder="EvaluationSoftware/ipsc-"+bright_field_num
    evaluation_path=os.path.join(data_folder,os.path.basename(res_path))
    deleteFile(evaluation_path)
    copyFile(res_path, evaluation_path)
    if isWin:
        evaluate_DET_command = "/EvaluationSoftware/Win/DETMeasure.exe" + " " + data_folder + \
                               " " + period_num + " " + "6"
        result, DET = subprocess.getstatusoutput(evaluate_DET_command)
        evaluate_TRA_command = "/EvaluationSoftware/Win/TRAMeasure.exe" + " " + data_folder + \
                               " " + period_num + " " + "6"
        result, TRA = subprocess.getstatusoutput(evaluate_TRA_command)
    else:
        evaluate_DET_command = "EvaluationSoftware/Linux/DETMeasure"+" " + data_folder +\
                               " "+period_num+" "+"6"
        result, DET = subprocess.getstatusoutput(evaluate_DET_command)
        evaluate_TRA_command = "EvaluationSoftware/Linux/TRAMeasure" +" "+ data_folder  +\
                               " " + period_num + " " + "6"
        result, TRA = subprocess.getstatusoutput(evaluate_TRA_command)
    DET=float(DET.replace("DET measure: ",""))
    TRA=float(TRA.replace("TRA measure: ",""))
    deleteFile(res_path)
    copyFile(evaluation_path,res_path)
    det_log_path=os.path.join(res_path,"DET_log.txt")
    with open(det_log_path, "r") as f:
        data = f.readlines()
    for index, line in enumerate(data):
        if "Splitting Operations" in line:
            num1 = index
        if "False Negative" in line:
            num2 = index
        if "False Positive" in line:
            num3 = index
    split = num2 - num1 - 1
    FN = num3 - num2 - 1
    FP = len(data) - num3 - 1

    track_list=os.listdir(res_path)
    track_list=[name for name in track_list if ".tif" in name]
    positive=0
    for name in track_list:
        mask=cv2.imread(os.path.join(res_path,name),-1)
        positive+=(len(np.unique(mask))-1)
    TP = positive - FP
    precision = TP / positive
    recall = TP / (TP + FN)
    F_measure = 2 * precision * recall / (precision + recall)
    FN_per_img = FN / (len(track_list))
    FP_per_img = FP / (len(track_list))

    '''GT_path=os.path.join(data_folder,period_num+"_GT/TRA")
    GT_list=os.listdir(GT_path)
    cell_num=0
    GT_list=[name for name in GT_list if ".tif" in name]
    for name in GT_list:
        GT=cv2.imread(os.path.join(GT_path,name),-1)
        cell_num+=(len(np.unique(GT))-1)
    cell_per_img=int(cell_num/len(GT_list))'''

    if period==3:
        print("======FN:{}  len(track_list):{}  FN_per_img:{}======".format(FN,len(track_list),FN_per_img))

    '''FN_per_img=str(FN_per_img)+"/"+str(cell_per_img)
    FP_per_img=str(FP_per_img)+"/"+str(cell_per_img)'''
    return DET,TRA,precision,recall,F_measure,FN_per_img,FP_per_img

def record_performance(excel_path,DET,TRA,precision,recall,F_measure,FN_per_img,FP_per_img):
    print("DET:",DET)
    print("TRA:",TRA)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-measure:",F_measure)
    book=xlwt.Workbook()
    DET_sheet=book.add_sheet('DET')
    TRA_sheet=book.add_sheet('TRA')
    pre_sheet=book.add_sheet('Precision')
    recall_sheet=book.add_sheet('Recall')
    f_sheet=book.add_sheet('F_measure')
    FN_sheet=book.add_sheet('FN_per_img')
    FP_sheet=book.add_sheet('FP_per_img')
    iteration_times=int(len(DET)/3)
    for i in range(iteration_times):
        DET_sheet.write(0,i+1,"iteration"+str(i+1).zfill(2))
        TRA_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        pre_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        recall_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        f_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        FN_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        FP_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        for k in range(3):
            DET_sheet.write(k+1,i+1,DET[i*3+k])
            TRA_sheet.write(k+1,i+1,TRA[i*3+k])
            pre_sheet.write(k + 1, i + 1, precision[i * 3 + k])
            recall_sheet.write(k + 1, i + 1, recall[i * 3 + k])
            f_sheet.write(k + 1, i + 1, F_measure[i * 3 + k])
            FN_sheet.write(k + 1, i + 1, FN_per_img[i * 3 + k])
            FP_sheet.write(k + 1, i + 1, FP_per_img[i * 3 + k])
    for j in range(3):
        DET_sheet.write(j+1,0,"period"+str(j+1).zfill(2))
        TRA_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        pre_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        recall_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        f_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        FN_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        FP_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))

    book.save(excel_path)
    print("\t\t\t performance has been recorded in ",excel_path)

def verify_and_evaluate(verify_bright_field_num,ckpt_path,result_dir):
    img_path = os.path.join(str(verify_bright_field_num) + "-GT", "imgs")
    unusual_path=os.path.join(str(verify_bright_field_num) + "-GT", "unusual")
    ckpts = os.listdir(ckpt_path)
    ckpts.sort()
    img_path_list=[]
    track_path_list=[]
    result_path_list=[]
    mask_without_unusual_path_list=[]
    for i in range(len(ckpts)):
        result_path=os.path.join(result_dir,ckpts[i].replace(".pth","")+"_result")
        predict_path = os.path.join(result_path, "predict")
        createFolder(result_path)
        createFolder(predict_path)
        print("verify using {}, saved in {}.".format(ckpt_path, predict_path))
        ckpt=os.path.join(ckpt_path, ckpts[i])
        test(img_path, predict_path, ckpt)
        track_path = os.path.join(result_path, "track")
        createFolder(track_path)
        #remove unusual
        mask_without_unusual_path=os.path.join(track_path,"mask")
        createFolder(mask_without_unusual_path)
        names = os.listdir(predict_path)
        names = [name for name in names if '.tif' in name]
        names.sort()
        unusual_list = os.listdir(unusual_path)
        unusual_list.sort()
        print("------removing unusual cells------")
        if len(names) != 233:
            names = [names[i] for i in range(len(names)) if i % 8 == 0]
        for i, name in enumerate(names):
            mask = cv2.imread(os.path.join(predict_path, name), -1)
            ret,mask=cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            unusual_mask = cv2.imread(os.path.join(unusual_path, unusual_list[i]), -1)
            unusual_mask = (unusual_mask > 0) * 1
            height = mask.shape[0]
            if height != 736:
                mask = mask[5: 741, 1: 769]
            ret, mask = cv2.connectedComponents(mask, ltype=2)
            candidates = np.unique(unusual_mask * mask)[1:]
            for candidate in candidates:
                unusual_label = (mask == candidate) * candidate
                unusual_label = unusual_label.astype(np.uint16)
                mask -= unusual_label
            mask = (mask != 0) * 255
            mask = mask.astype(np.uint8)
            cv2.imwrite(os.path.join(mask_without_unusual_path, str(i).zfill(6) + ".tif"), mask)
        track_path_list.append(track_path)
        img_path_list.append(img_path)
        mask_without_unusual_path_list.append(mask_without_unusual_path)
        result_path_list.append(result_path)
    with Pool() as p:
        p.map(track_all_periods,img_path_list,mask_without_unusual_path_list,track_path_list)

    DET_result = []
    TRA_result = []
    precision = []
    recall = []
    F_measure = []
    FN_per_img = []
    FP_per_img = []
    for folder in result_path_list:
        track_path=os.path.join(folder,"track")
        for i in range(1, 4):
            DET, TRA, pre, cal, fm, fnpi, fppi = evaluate_DET_TRA(bright_field=verify_bright_field_num, period=i, track_result_path=track_path)
            DET_result.append(DET)
            TRA_result.append(TRA)
            precision.append(pre)
            recall.append(cal)
            F_measure.append(fm)
            FN_per_img.append(fnpi)
            FP_per_img.append(fppi)
    result_excel = os.path.join(result_dir, "verify_result.xls")
    record_performance(result_excel, DET_result, TRA_result, precision, recall, F_measure, FN_per_img, FP_per_img)

def track_with_pool(img_path,folder_path,unusual_path):
    print(folder_path)
    mask_path = os.path.join(folder_path, "mask")
    print(mask_path)
    track_path = os.path.join(folder_path, "track")
    createFolder(track_path)
    mask_without_unusual_path=os.path.join(track_path,"mask")
    createFolder(mask_without_unusual_path,clean=True)
    #去除异常情况
    names = os.listdir(mask_path)
    names = [name for name in names if '.tif' in name]
    names.sort()
    unusual_list=os.listdir(unusual_path)
    unusual_list.sort()
    print("------removing unusual cells------")
    if len(names) != 233:
        names = [names[i] for i in range(len(names)) if i % 8 == 0]
    for i,name in enumerate(names):
        mask=cv2.imread(os.path.join(mask_path,name),-1)
        unusual_mask=cv2.imread(os.path.join(unusual_path,unusual_list[i]),-1)
        unusual_mask=(unusual_mask>0)*1
        height = mask.shape[0]
        if height!=736:
            mask=mask[5: 741, 1: 769]

        ret, mask = cv2.connectedComponents(mask,ltype=2)
        candidates=np.unique(unusual_mask*mask)[1:]
        for candidate in candidates:
            unusual_label=(mask==candidate)*candidate
            unusual_label=unusual_label.astype(np.uint16)
            mask-=unusual_label
        mask = (mask != 0) * 255
        mask = mask.astype(np.uint8)
        cv2.imwrite(os.path.join(mask_without_unusual_path,str(i).zfill(6)+".tif"),mask)
    track_all_periods(img_path, mask_without_unusual_path, track_path)

def iteration_and_evaluate(bright_field_num,iteration_times,keep_training,verify_BF_num,batch_size=32,shuffle=False):
    img_path=str(bright_field_num)+"-GT/imgs"
    mask_path=str(bright_field_num)+"-GT/flore-mask"
    unusual_path=str(bright_field_num)+"-GT/unusual"
    iteration_path=str(bright_field_num)+"-Iteration"
    img_aug_path = os.path.join(iteration_path, "imgs_aug")
    first_iteration_path = os.path.join(iteration_path, "iteration01")
    mask_aug_path = os.path.join(first_iteration_path, "mask")
    createFolder(iteration_path)
    createFolder(img_aug_path)
    createFolder(first_iteration_path)
    createFolder(mask_aug_path)

    augmentationWithPool(img_path, img_aug_path, mask_path, mask_aug_path, mask=True)
    trainWithIteration(img_path,img_aug_path, iteration_path, keep_training, iteration_times,batch_size,shuffle)

    iteration_folder_list = os.listdir(iteration_path)
    iteration_folder_list=[name for name in iteration_folder_list if "iteration" in name]
    iteration_folder_list.sort()
    print(iteration_folder_list)

    img_path_list = []
    unusual_path_list=[]
    for i in range(len(iteration_folder_list)):
        img_path_list.append(img_path)
        unusual_path_list.append(unusual_path)
    folder_path_list=[os.path.join(iteration_path,name) for name in iteration_folder_list]
    with Pool() as p:
        p.map(track_with_pool,img_path_list,folder_path_list,unusual_path_list)

    DET_result = []
    TRA_result = []
    precision=[]
    recall=[]
    F_measure=[]
    FN_per_img=[]
    FP_per_img=[]
    for folder in iteration_folder_list:
        folder_path = os.path.join(iteration_path, folder)
        track_path = os.path.join(folder_path, "track")
        for i in range(1, 4):
            DET, TRA,pre,cal,fm,fnpi,fppi = evaluate_DET_TRA(bright_field=bright_field_num, period=i, track_result_path=track_path)
            DET_result.append(DET)
            TRA_result.append(TRA)
            precision.append(pre)
            recall.append(cal)
            F_measure.append(fm)
            FN_per_img.append(fnpi)
            FP_per_img.append(fppi)
    result_excel = os.path.join(iteration_path, "evaluation_result.xls")
    record_performance(result_excel, DET_result, TRA_result,precision,recall,F_measure,FN_per_img,FP_per_img)

    print("==========Strating verfify==========")
    verfiy_result_path=os.path.join(iteration_path,"verify")
    createFolder(verfiy_result_path)
    checkpoint_path=os.path.join(iteration_path,"checkpoints")
    verify_and_evaluate(verify_BF_num,checkpoint_path,verfiy_result_path)


if __name__=="__main__":
    iteration_times=10
    keep_training=True
    shuffle=True
    batch_size=8
    train_field=7
    test_field=2
    iteration_and_evaluate(train_field,iteration_times,keep_training,test_field,batch_size,shuffle)
    '''
    train_field = 2
    test_field = 7
    iteration_and_evaluate(train_field, iteration_times, keep_training, test_field, batch_size, shuffle)'''



# Incomplete initial labeling, Ground Truth, Train and Test Models

All scripts are given inside this folder to perform the specified tasks. 

## Incomplete initial labeling of brightfield images using the paired red fluorescent images

```process_image``` folder contains the sample of preprocessed brightfield and red fluorescent data. These data are used to create incomplete initial labels. To get incomplete initial labels run the following python script and adjust the parameters:

```
python process_mask.py
```

## Ground truth labels of brightfield cell images

```draw_mask``` folder contains the results of processing the brightfield and red fluorescent data from ```process_image``` folder. The python file named ```draw.py``` inside the folder ```draw_mask``` represents a program based on OpenCV and red fluorescent image developed to process and annotate cells manually in brightfield images. The brightfield image was superimposed on the paired red fluorescent image with the help of an interactive window, and the paired red fluorescent image provides a reference for cell labeling. The left mouse button is used to draw a mask on the fluorescent image, while the right mouse button is used to erase the mask.
Run the following script to perform the task described here: 

```
python draw.py
```

## Training and testing


```train``` folder contains the whole implementation process of the project.

Run the following script to get the incomplete labels/unusual used to start the training process: 

```
python remove_mask.py
```

To train and test the proposed model run the script: 

```
python run_with_iteration.py
```

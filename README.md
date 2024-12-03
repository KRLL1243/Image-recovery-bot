# U-Net neural network for restoring blurred images

## 1. Dataset
### The model was trained on the [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
### Examples:
<img src="images/dataset_examples/000002.jpg" width="178" height="218"/> <img src="images/dataset_examples/000004.jpg" width="178" height="218"/> <img src="images/dataset_examples/000080.jpg" width="178" height="218"/>

## 2. Data preprocessing
### Script for data preprocessing: `dataset_preparing.py`
### Input:
<img src="images/dataset_examples/000080.jpg" width="178" height="218"/>

### Output:
<img src="images/preprocessed_dataset_examples/000080.jpg" width="512" height="256"/>

## 3. Training model
### Loss Graphic:
![loss.png](images/loss.png)

## 4. Testing model(dataset images)
### Results:
![pic1.png](images/test_results/pic1.png)
![pic2.png](images/test_results/pic2.png)
![pic3.png](images/test_results/pic3.png)
![pic4.png](images/test_results/pic4.png)
![pic5.png](images/test_results/pic5.png)
![pic6.png](images/test_results/pic6.png)
![pic7.png](images/test_results/pic7.png)
![pic8.png](images/test_results/pic8.png)
![pic9.png](images/test_results/pic9.png)
![pic10.png](images/test_results/pic10.png)

## 5. Testing model(NOT dataset images)
### Script for testing on new image: `test_unet.py`
### Results:

![not_dataset_pic1.png](images/test_results/not_dataset_pic1.png)

![not_dataset_pic2.png](images/test_results/not_dataset_pic2.png)

ㅤ
### Trained model: `unet_model.pth`
#### How to load:
```
model = UNet().to(device).eval()
model.load_state_dict(torch.load('unet_model.pth', map_location=device, weights_only=True))
```

ㅤ
## 6. Model integration into telegram bot
### Scripts for creating a telegram bot: `tg_bot.py`,`image_recovery.py`

### Example ot using:
<img src="images/tg_screen_1.png"/>
<img src="images/tg_screen_2.png"/>

#### [Bot](https://t.me/ImageRecoveryBot)
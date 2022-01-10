# Real-Time Facial Emotion Detection

### Description
- This Repository contains the code files for building a Deep Convolutional Neural Netwok based model to identify the human facial emotions into 5 classes namely Neutral, Happy, Sad, Fear and Angry.
- It can be able to detect and classify the facial emotions in real time also. 

### Predictions
![happy](https://user-images.githubusercontent.com/71257512/148689048-db1d759e-93cb-4dbf-81f9-667c441ec134.jpg)
![sad1](https://user-images.githubusercontent.com/71257512/148688620-aa18a674-77c1-4ef6-b5df-729ed746b762.jpg)

![fear2](https://user-images.githubusercontent.com/71257512/148688756-aab609b3-c363-4199-b6c4-9cdd952ddf17.jpg)
![neutral2](https://user-images.githubusercontent.com/71257512/148688871-56f79614-082f-42bb-9c1d-0ecb7d9ee00b.jpg)

### Dataset
- The FER2013 dataset consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
- The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.
- The Dataset was sourced from this [link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
![fer](https://user-images.githubusercontent.com/71257512/148721671-c33e60e7-a14b-455f-b39d-da154c4b1f30.png)

### Training
- The Model was trained using Resnet-152 architecture by the transfer learning mechanism.
- It was trained for 30 epochs.

### Tools Used
- Python 
- Pytorch
- OpenCV

Due to limited Storage, I didn't include the model in this repository. 

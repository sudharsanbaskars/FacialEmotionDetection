import torch
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_model(path):
    return torch.load(path)

def detect_emotions(img_path, model):
    
    emotion_dict = {0: "angry", 1: "fear", 2:"happy", 3: "sad", 4: "neutral"}
    

    val_transform = transforms.Compose([
        transforms.ToTensor()])


    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_face.xml')
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        X = resize_frame/256
        X = Image.fromarray((resize_frame))
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = emotion_dict[int(top_class.numpy())]
        cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255), 2)
        print(pred)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    img_path = "images/sad.jfif"
    path = "models/model.pth"
    model = load_model(path)
    detect_emotions(img_path, model)
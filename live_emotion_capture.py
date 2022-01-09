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

def detect_emotions_using_webcam():
    path = "models/model.pth"
    model = load_model(path)
    emotion_dict = {0: "angry", 1: "fear", 2:"happy", 3: "sad", 4: "neutral"}


    val_transform = transforms.Compose([
        transforms.ToTensor()])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('./models/haarcascade_face.xml')
        faces = face_cascade.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            X = resize_frame/256
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = emotion_dict[int(top_class.numpy())]
            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255), 3)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_using_webcam()
import os
import cv2
import numpy as np
import requests
from pathlib import Path

# Create data and outputs folders if they don't exist
Path('data').mkdir(parents=True, exist_ok=True)
Path('outputs').mkdir(parents=True, exist_ok=True)

# URLs for public-domain images
IMAGES = {
    "image1.jpg": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=800",  # full-body portrait
    "image2.jpg": "https://images.unsplash.com/photo-1484933653696-477063a600c8?w=800",  # crowd photo
    "image3.jpg": "https://www.publicdomainpictures.net/pictures/300000/nahled/farm-animals.jpg"  # animal
}

# Download images if not already present
for filename, url in IMAGES.items():
    path = os.path.join("data", filename)
    if not os.path.isfile(path):
        print(f"Downloading {filename}...")
        resp = requests.get(url)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            f.write(resp.content)
        print("Downloaded.")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_and_blur_eyes(image_path, out_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        face = img[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(15,15))
        for (ex,ey,ew,eh) in eyes:
            eye_roi = face[ey:ey+eh, ex:ex+ew]
            face[ey:ey+eh, ex:ex+ew] = cv2.GaussianBlur(eye_roi, (21,21), 30)

    cv2.imwrite(out_path, img)

# Process downloaded images and save outputs to outputs folder
for imgfile in IMAGES:
    inp = f"data/{imgfile}"
    out = f"outputs/output_{imgfile}"
    print(f"Processing {inp} → {out}")
    detect_and_blur_eyes(inp, out)

print("All done!")

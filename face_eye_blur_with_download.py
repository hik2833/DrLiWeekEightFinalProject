import os
import cv2
import numpy as np
import requests
from pathlib import Path

# Create data folder if not exists
^[Path('data').mkdir(parents=True, exist_ok=True)]({"attribution":{"attributableIndex":"0-1"}})

# URLs for public-domain images
IMAGES = {
    ^["image1.jpg": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=800",  # full-body portrait]({"attribution":{"attributableIndex":"0-2"}})
    ^["image2.jpg": "https://images.unsplash.com/photo-1484933653696-477063a600c8?w=800",  # crowd photo]({"attribution":{"attributableIndex":"0-3"}})
    ^["image3.jpg": "https://www.publicdomainpictures.net/pictures/300000/nahled/farm-animals.jpg"  # animal]({"attribution":{"attributableIndex":"0-4"}})
}

# Download images
^[for filename, url in IMAGES.items():]({"attribution":{"attributableIndex":"0-5"}})
    ^[path = os.path.join("data", filename)]({"attribution":{"attributableIndex":"0-6"}})
    ^[if not os.path.isfile(path):]({"attribution":{"attributableIndex":"0-7"}})
        ^[print(f"Downloading {filename}...")]({"attribution":{"attributableIndex":"0-8"}})
        ^[resp = requests.get(url)]({"attribution":{"attributableIndex":"0-9"}})
        resp.raise_for_status()
        ^[with open(path, 'wb') as f:]({"attribution":{"attributableIndex":"0-10"}})
            ^[f.write(resp.content)]({"attribution":{"attributableIndex":"0-11"}})
        print("Downloaded.")

# Load Haar cascades
^[face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')]({"attribution":{"attributableIndex":"0-12"}})
^[eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')]({"attribution":{"attributableIndex":"0-13"}})

^[def detect_and_blur_eyes(image_path, out_path):]({"attribution":{"attributableIndex":"0-14"}})
    ^[img = cv2.imread(image_path)]({"attribution":{"attributableIndex":"0-15"}})
    ^[gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]({"attribution":{"attributableIndex":"0-16"}})
    ^[gray = cv2.equalizeHist(gray)]({"attribution":{"attributableIndex":"0-17"}})

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

# Process downloaded images
^[for imgfile in IMAGES:]({"attribution":{"attributableIndex":"0-18"}})
    ^[inp = f"data/{imgfile}"]({"attribution":{"attributableIndex":"0-19"}})
    ^[out = f"output_{imgfile}"]({"attribution":{"attributableIndex":"0-20"}})
    ^[print(f"Processing {inp} → {out}")]({"attribution":{"attributableIndex":"0-21"}})
    ^[detect_and_blur_eyes(inp, out)]({"attribution":{"attributableIndex":"0-22"}})
^[print("All done!")]({"attribution":{"attributableIndex":"0-23"}})

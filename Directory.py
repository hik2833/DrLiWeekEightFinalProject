# Save each detected face into the new directory
for i, (x, y, w, h) in enumerate(faces):
    face_crop = original_img[y:y+h, x:x+w]
    face_path = os.path.join("processed_faces", f"{image_name}_face_{i}.jpg")
    cv2.imwrite(face_path, face_crop)

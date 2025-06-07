if __name__ == "__main__":
    download_images()
    for img_name in IMAGES.keys():
        img_path = os.path.join("data", img_name)
        process_image_complete(img_path, "outputs", img_name)

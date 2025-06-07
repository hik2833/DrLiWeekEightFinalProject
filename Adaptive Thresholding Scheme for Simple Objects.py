import os
import cv2
import numpy as np
import requests
from pathlib import Path
import matplotlib.pyplot as plt

# Create data and outputs folders if they don't exist
Path('data').mkdir(parents=True, exist_ok=True)
Path('outputs').mkdir(parents=True, exist_ok=True)

# URLs for the three required image types
IMAGES = {
    "indoor_scene.jpg": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800",  # Indoor living room
    "outdoor_scenery.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",  # Mountain landscape  
    "close_up_object.jpg": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=800"  # Single flower close-up
}

def download_images():
    """Download images if not already present"""
    for filename, url in IMAGES.items():
        path = os.path.join("data", filename)
        if not os.path.isfile(path):
            print(f"Downloading {filename}...")
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(resp.content)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

def otsu_threshold(image):
    """
    Otsu's automatic threshold selection method
    Finds optimal threshold by maximizing between-class variance
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # Total number of pixels
    total = image.shape[0] * image.shape[1]
    
    # Initialize variables for Otsu's method
    sum_total = sum(i * hist[i] for i in range(256))
    sum_background = 0
    weight_background = 0
    weight_foreground = 0
    variance_max = 0
    threshold_optimal = 0
    
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
            
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += t * hist[t]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Between-class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > variance_max:
            variance_max = variance_between
            threshold_optimal = t
    
    return threshold_optimal

def adaptive_mean_threshold(image, block_size=11, C=2):
    """
    Adaptive mean thresholding - threshold is mean of neighborhood minus constant C
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, block_size, C)

def adaptive_gaussian_threshold(image, block_size=11, C=2):
    """
    Adaptive Gaussian thresholding - threshold is Gaussian-weighted sum of neighborhood minus C
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, block_size, C)

def local_threshold_niblack(image, window_size=15, k=0.2):
    """
    Niblack's local thresholding method
    T = mean + k * standard_deviation
    """
    # Pad image to handle borders
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # Initialize output
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract local window
            window = padded[i:i+window_size, j:j+window_size]
            
            # Calculate local statistics
            mean_local = np.mean(window)
            std_local = np.std(window)
            
            # Niblack threshold
            threshold = mean_local + k * std_local
            
            # Apply threshold
            result[i, j] = 255 if image[i, j] > threshold else 0
    
    return result

def sauvola_threshold(image, window_size=15, k=0.5, R=128):
    """
    Sauvola's thresholding method (improvement over Niblack)
    T = mean * (1 + k * ((std/R) - 1))
    """
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            mean_local = np.mean(window)
            std_local = np.std(window)
            
            # Sauvola threshold
            threshold = mean_local * (1 + k * ((std_local / R) - 1))
            result[i, j] = 255 if image[i, j] > threshold else 0
    
    return result

def multi_level_otsu(image, num_thresholds=2):
    """
    Multi-level Otsu thresholding for multiple regions
    """
    # Use OpenCV's built-in multi-threshold Otsu
    thresholds = []
    
    # For simplicity, we'll use a recursive approach for 2-level
    if num_thresholds == 2:
        # First threshold using standard Otsu
        thresh1 = otsu_threshold(image)
        
        # Split image into two regions
        lower_region = image[image <= thresh1]
        upper_region = image[image > thresh1]
        
        # Find second threshold in upper region
        if len(upper_region) > 0:
            # Create temporary image for upper region
            temp_image = np.where(image > thresh1, image, 0)
            thresh2 = otsu_threshold(temp_image)
            thresholds = [thresh1, thresh2]
        else:
            thresholds = [thresh1]
    
    return thresholds

def process_image_with_all_methods(image_path, image_name):
    """
    Apply all adaptive thresholding methods to an image
    """
    print(f"\nProcessing {image_name}...")
    
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Apply preprocessing to reduce noise
    img_denoised = cv2.GaussianBlur(img, (3, 3), 0)
    
    print(f"Image dimensions: {img.shape}")
    print(f"Mean intensity: {np.mean(img):.2f}")
    print(f"Standard deviation: {np.std(img):.2f}")
    
    # Method 1: Otsu's thresholding
    otsu_thresh = otsu_threshold(img_denoised)
    _, otsu_result = cv2.threshold(img_denoised, otsu_thresh, 255, cv2.THRESH_BINARY)
    print(f"Otsu threshold: {otsu_thresh}")
    
    # Method 2: Adaptive Mean
    adaptive_mean = adaptive_mean_threshold(img_denoised)
    
    # Method 3: Adaptive Gaussian
    adaptive_gaussian = adaptive_gaussian_threshold(img_denoised)
    
    # Method 4: Niblack's method
    niblack_result = local_threshold_niblack(img_denoised)
    
    # Method 5: Sauvola's method
    sauvola_result = sauvola_threshold(img_denoised)
    
    # Save all results
    base_name = image_name.replace('.jpg', '')
    cv2.imwrite(f'outputs/{base_name}_original.jpg', img)
    cv2.imwrite(f'outputs/{base_name}_otsu.jpg', otsu_result)
    cv2.imwrite(f'outputs/{base_name}_adaptive_mean.jpg', adaptive_mean)
    cv2.imwrite(f'outputs/{base_name}_adaptive_gaussian.jpg', adaptive_gaussian)
    cv2.imwrite(f'outputs/{base_name}_niblack.jpg', niblack_result)
    cv2.imwrite(f'outputs/{base_name}_sauvola.jpg', sauvola_result)
    
    # Create comparison plot
    create_comparison_plot(img, otsu_result, adaptive_mean, adaptive_gaussian, 
                          niblack_result, sauvola_result, base_name)
    
    print(f"Results saved for {image_name}")

def create_comparison_plot(original, otsu, adaptive_mean, adaptive_gaussian, niblack, sauvola, base_name):
    """
    Create a comparison plot showing all thresholding results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Adaptive Thresholding Results - {base_name}', fontsize=16)
    
    images = [original, otsu, adaptive_mean, adaptive_gaussian, niblack, sauvola]
    titles = ['Original', 'Otsu', 'Adaptive Mean', 'Adaptive Gaussian', 'Niblack', 'Sauvola']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row, col = i // 3, i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'outputs/{base_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_segmentation_quality(image_path, image_name):
    """
    Analyze the quality of segmentation for different methods
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_denoised = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply different methods
    otsu_thresh = otsu_threshold(img_denoised)
    _, otsu_result = cv2.threshold(img_denoised, otsu_thresh, 255, cv2.THRESH_BINARY)
    adaptive_gaussian = adaptive_gaussian_threshold(img_denoised)
    
    # Calculate metrics
    print(f"\nSegmentation Analysis for {image_name}:")
    
    # Foreground/background ratio
    otsu_fg_ratio = np.sum(otsu_result == 255) / otsu_result.size
    adaptive_fg_ratio = np.sum(adaptive_gaussian == 255) / adaptive_gaussian.size
    
    print(f"Otsu - Foreground ratio: {otsu_fg_ratio:.3f}")
    print(f"Adaptive Gaussian - Foreground ratio: {adaptive_fg_ratio:.3f}")
    
    # Edge preservation (using Canny edge detection)
    edges_original = cv2.Canny(img_denoised, 50, 150)
    edges_otsu = cv2.Canny(otsu_result, 50, 150)
    edges_adaptive = cv2.Canny(adaptive_gaussian, 50, 150)
    
    edge_preservation_otsu = np.sum(edges_otsu) / np.sum(edges_original) if np.sum(edges_original) > 0 else 0
    edge_preservation_adaptive = np.sum(edges_adaptive) / np.sum(edges_original) if np.sum(edges_original) > 0 else 0
    
    print(f"Otsu - Edge preservation: {edge_preservation_otsu:.3f}")
    print(f"Adaptive Gaussian - Edge preservation: {edge_preservation_adaptive:.3f}")

def main():
    """
    Main function to run the complete adaptive thresholding analysis
    """
    print("=== Adaptive Thresholding for Image Segmentation ===")
    print("This program implements multiple adaptive thresholding techniques")
    print("to segment three types of images: indoor, outdoor, and close-up scenes.\n")
    
    # Download images
    download_images()
    
    # Process each image with all methods
    for filename in IMAGES.keys():
        image_path = os.path.join("data", filename)
        if os.path.exists(image_path):
            process_image_with_all_methods(image_path, filename)
            analyze_segmentation_quality(image_path, filename)
        else:
            print(f"Warning: {image_path} not found")
    
    print("\n=== Summary ===")
    print("All images have been processed with multiple adaptive thresholding methods:")
    print("1. Otsu's method - Global optimal threshold")
    print("2. Adaptive Mean - Local mean-based threshold")
    print("3. Adaptive Gaussian - Gaussian-weighted local threshold")
    print("4. Niblack's method - Mean + k*std deviation")
    print("5. Sauvola's method - Improved Niblack for text images")
    print("\nResults saved in 'outputs' folder with comparison plots.")
    print("Each method automatically selects thresholds without human intervention.")

if __name__ == "__main__":
    main()

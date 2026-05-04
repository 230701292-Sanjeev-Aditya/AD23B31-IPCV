import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: DATA COLLECTION & ORGANIZATION
# ==========================================
# The script expects the 'archive' folder in the same directory
UPLOADED_ROOT = '.' 
BASE_DIR = 'wildlife_project'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

def setup_local_project():
    if not os.path.exists(UPLOADED_ROOT):
        print(f"Error: Folder '{UPLOADED_ROOT}' not found! Ensure 'archive' is next to main.py.")
        return False

    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    # Buffalo, elephant, rhino, and zebra folders from your archive
    classes = ['zebra', 'elephant', 'rhino', 'buffalo']
    
    for cls in classes:
        src_path = os.path.join(UPLOADED_ROOT, cls)
        if os.path.exists(src_path):
            dst_path = os.path.join(DATASET_DIR, f'class_{cls}')
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            
            # Filter: Keep images (001.jpg) and ignore text files (001.txt)
            all_files = os.listdir(dst_path)
            images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Remove non-image files to clean the dataset
            for f in all_files:
                if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    os.remove(os.path.join(dst_path, f))
            
            print(f"Step 1: Imported {len(images)} images for class_{cls}")

            # Automatically set the first image as the Species Template[cite: 1]
            if images:
                shutil.copy(os.path.join(dst_path, images[0]), 
                            os.path.join(TEMPLATE_DIR, f"{cls}_template.jpg"))
    return True

# ==========================================
# STEP 2: MANDATORY PREPROCESSING[cite: 1]
# ==========================================
def apply_preprocessing(img):
    """Applies Resize, Denoise, and Enhance filters[cite: 1]"""
    # 1. Resize: Standardize all images to 224x224[cite: 1]
    img = cv2.resize(img, (224, 224))
    
    # 2. Denoise: Remove noise using Gaussian Blur[cite: 1]
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    
    # 3. Enhance: Improve contrast using CLAHE[cite: 1]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img

# ==========================================
# STEP 3 & 4: FEATURE EXTRACTION (ORB) & ALGORITHM[cite: 1]
# ==========================================
def identify_species_orb(template_path, scene_path):
    # Load and Preprocess[cite: 1]
    img_t = apply_preprocessing(cv2.imread(template_path))
    img_s = apply_preprocessing(cv2.imread(scene_path))
    
    # Convert to grayscale for ORB feature extraction[cite: 1]
    gray_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector to extract feature vectors[cite: 1]
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_t, None)
    kp2, des2 = orb.detectAndCompute(gray_s, None)

    # Brute-Force Matcher for binary descriptors[cite: 1]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is not None and des2 is not None:
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        # Visual output showing feature matches[cite: 1]
        result_viz = cv2.drawMatches(gray_t, kp1, gray_s, kp2, matches[:15], None, flags=2)
        return len(matches), result_viz
    return 0, None

# ==========================================
# STEP 5: EVALUATION[cite: 1]
# ==========================================
def run_evaluation():
    species_list = ['zebra', 'elephant', 'rhino', 'buffalo']
    print("\n--- Starting Algorithm Evaluation ---")
    
    for species in species_list:
        template = os.path.join(TEMPLATE_DIR, f"{species}_template.jpg")
        class_folder = os.path.join(DATASET_DIR, f"class_{species}")
        
        if not os.path.exists(template): continue
        
        # Test on a subset for the performance table[cite: 1]
        test_images = os.listdir(class_folder)[:10]
        match_scores = []
        
        for i, img_name in enumerate(test_images):
            count, viz = identify_species_orb(template, os.path.join(class_folder, img_name))
            match_scores.append(count)
            
            # Show first match window for your Demo Video/Screenshots[cite: 1]
            if i == 0 and viz is not None:
                cv2.imshow(f"ORB Species ID: {species.capitalize()}", viz)
                cv2.waitKey(2000) # Displays for 2 seconds
        
        # Calculate Accuracy based on matching threshold[cite: 1]
        accuracy = (sum(1 for s in match_scores if s > 40) / len(test_images)) * 100
        print(f"Final Accuracy for {species}: {accuracy}%[cite: 1]")
        
    cv2.destroyAllWindows()

# Main execution loop[cite: 1]
if __name__ == "__main__":
    if setup_local_project():
        run_evaluation()

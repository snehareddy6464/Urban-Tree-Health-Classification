import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['damaged', 'dead', 'diseased', 'healthy']
SAMPLES_PER_CLASS = 100
IMG_SIZE = (224, 224)

def generate_healthy_sample():
    """
    Generates realistic healthy plant samples:
    - Close-up leaf macro shots (various greens, clear leaf veins, uniform chlorophyll)
    - Full tree canopy (vibrant green, natural soil/sky backgrounds)
    """
    sample_type = random.choice(['leaf_macro', 'tree_canopy', 'leaf_cluster'])
    bg_r = random.randint(20, 100)
    bg_g = random.randint(30, 120)
    bg_b = random.randint(20, 100)
    img = Image.new("RGB", IMG_SIZE, (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    
    if sample_type == 'leaf_macro':
        # Close-up green leaf (all natural shades: dark forest green to bright emerald)
        leaf_g = random.randint(70, 245)
        leaf_r = random.randint(10, min(leaf_g - 15, 140))
        leaf_b = random.randint(10, min(leaf_g - 15, 110))
        
        # Leaf shape
        draw.ellipse([20, 15, 204, 209], fill=(leaf_r, leaf_g, leaf_b))
        
        # Leaf midrib & veins
        vein_color = (max(0, leaf_r - 20), max(0, leaf_g - 35), max(0, leaf_b - 15))
        draw.line([(112, 20), (112, 200)], fill=vein_color, width=3)
        for y in range(40, 180, 25):
            draw.line([(112, y), (60, y - 20)], fill=vein_color, width=2)
            draw.line([(112, y), (164, y - 20)], fill=vein_color, width=2)
            
    elif sample_type == 'tree_canopy':
        # Tree canopy with trunk
        draw.rectangle([95, 130, 129, 224], fill=(80, 50, 30))
        np_img = np.array(img, dtype=np.uint8)
        for _ in range(45):
            cx = random.randint(30, 194)
            cy = random.randint(20, 150)
            radius = random.randint(20, 60)
            g_val = random.randint(140, 230)
            r_val = random.randint(15, 85)
            b_val = random.randint(15, 75)
            
            y_indices, x_indices = np.ogrid[:224, :224]
            mask = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2) <= radius
            np_img[mask] = [r_val, g_val, b_val]
        img = Image.fromarray(np_img)
        
    else:
        # Cluster of healthy leaves
        for _ in range(8):
            lx = random.randint(40, 180)
            ly = random.randint(40, 180)
            rx = random.randint(25, 55)
            ry = random.randint(35, 75)
            draw.ellipse([lx - rx, ly - ry, lx + rx, ly + ry], fill=(random.randint(25, 90), random.randint(150, 235), random.randint(20, 80)))

    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))

def generate_diseased_sample():
    """
    Generates realistic diseased plant samples:
    - Chlorosis (yellowing/browning foliage)
    - Necrotic spots, leaf blight, black/brown lesion clusters, mold patches
    """
    img = Image.new("RGB", IMG_SIZE, (random.randint(30, 90), random.randint(30, 90), random.randint(30, 70)))
    draw = ImageDraw.Draw(img)
    
    # Diseased leaf surface (chlorotic yellowing / sickly brown-green)
    leaf_r = random.randint(150, 220)
    leaf_g = random.randint(140, 200)
    leaf_b = random.randint(15, 70)
    draw.ellipse([20, 15, 204, 209], fill=(leaf_r, leaf_g, leaf_b))
    
    # Midrib
    draw.line([(112, 20), (112, 200)], fill=(110, 100, 30), width=3)
    
    np_img = np.array(img, dtype=np.uint8)
    
    # Necrotic dark brown/black spot clusters (fungal lesions & blight)
    num_spots = random.randint(15, 45)
    for _ in range(num_spots):
        sx = random.randint(35, 185)
        sy = random.randint(30, 190)
        sr = random.randint(4, 16)
        
        spot_r = random.randint(40, 90)
        spot_g = random.randint(25, 55)
        spot_b = random.randint(10, 35)
        
        y_indices, x_indices = np.ogrid[:224, :224]
        mask = np.sqrt((x_indices - sx)**2 + (y_indices - sy)**2) <= sr
        np_img[mask] = [spot_r, spot_g, spot_b]
        
    return Image.fromarray(np_img)

def generate_damaged_sample():
    """
    Generates realistic damaged tree samples:
    - Deep trunk split cracks, bark cavities, broken limbs, stripped bark
    """
    img = Image.new("RGB", IMG_SIZE, (random.randint(100, 160), random.randint(130, 180), random.randint(160, 210)))
    draw = ImageDraw.Draw(img)
    
    # Ground & background
    draw.rectangle([0, 175, 224, 224], fill=(85, 105, 65))
    
    # Trunk
    draw.rectangle([75, 50, 149, 224], fill=(85, 60, 40))
    
    # Split crack / dark cavity
    draw.polygon([(102, 70), (128, 65), (122, 195), (105, 205)], fill=(20, 12, 8))
    draw.polygon([(110, 80), (122, 75), (116, 185)], fill=(180, 140, 90)) # Exposed inner wood
    
    # Broken limb
    draw.polygon([(149, 105), (195, 75), (190, 90), (149, 125)], fill=(70, 45, 30))
    draw.polygon([(190, 75), (200, 80), (193, 93)], fill=(35, 20, 10))
    
    # Sparse canopy
    np_img = np.array(img, dtype=np.uint8)
    for _ in range(12):
        cx = random.randint(50, 170)
        cy = random.randint(25, 75)
        radius = random.randint(12, 30)
        y_indices, x_indices = np.ogrid[:224, :224]
        mask = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2) <= radius
        np_img[mask] = [random.randint(70, 110), random.randint(100, 140), random.randint(35, 65)]
        
    return Image.fromarray(np_img)

def generate_dead_sample():
    """
    Generates realistic dead tree/plant samples:
    - Dry leafless brown/gray branches, zero green chlorophyll, withered brittle wood
    """
    img = Image.new("RGB", IMG_SIZE, (random.randint(140, 180), random.randint(160, 190), random.randint(170, 205)))
    draw = ImageDraw.Draw(img)
    
    # Dry ground
    draw.rectangle([0, 180, 224, 224], fill=(125, 115, 95))
    
    # Bare dry trunk
    draw.polygon([(98, 65), (126, 65), (132, 224), (92, 224)], fill=(65, 55, 45))
    
    # Bare brittle dry branches
    branches = [
        ([(112, 115), (55, 65), (35, 45)], 5),
        ([(112, 115), (45, 85), (25, 80)], 4),
        ([(112, 95), (165, 45), (195, 30)], 5),
        ([(112, 95), (175, 75), (205, 70)], 4),
        ([(112, 75), (98, 25), (92, 10)], 4),
        ([(112, 75), (132, 25), (148, 8)], 4)
    ]
    for pts, width in branches:
        draw.line(pts, fill=(55, 45, 35), width=width)
        
    return img

def create_dataset():
    print("Generating tree health training dataset...")
    generators = {
        'healthy': generate_healthy_sample,
        'diseased': generate_diseased_sample,
        'damaged': generate_damaged_sample,
        'dead': generate_dead_sample
    }
    
    total_created = 0
    for cls_name, gen_func in generators.items():
        cls_dir = os.path.join(BASE_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Clear existing images
        for f in os.listdir(cls_dir):
            if f.endswith(('.jpg', '.png')):
                os.remove(os.path.join(cls_dir, f))
                
        for i in range(SAMPLES_PER_CLASS):
            img = gen_func()
            img_path = os.path.join(cls_dir, f"{cls_name}_{i+1:03d}.jpg")
            img.save(img_path, quality=95)
            total_created += 1
            
        print(f"Generated {SAMPLES_PER_CLASS} images for class '{cls_name}'")
        
    print(f"Dataset generation complete! Total images created: {total_created}")

if __name__ == "__main__":
    create_dataset()


import os
import io
import numpy as np
from PIL import Image
try:
    import keras
except ImportError:
    from tensorflow import keras
import tensorflow as tf

CLASSES = ['Damaged', 'Dead', 'Diseased', 'Healthy']

# Path to saved model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'tree_health_mobilenetv2.h5')

# Global model cache
# Global model cache and timestamp
_model = None
_model_mtime = 0

RECOMMENDATIONS = {
    'Healthy': "The tree displays vibrant foliage and sturdy bark. Maintain current care schedule including weekly watering during dry spells, annual soil fertilization, and light crown pruning to preserve balanced airflow.",
    'Diseased': "Symptoms of disease detected (such as leaf spotting, blight, or fungal growth). Isolate affected foliage, prune dead twigs with sanitized shears, improve root ventilation, and apply targeted organic fungicide/bactericide.",
    'Damaged': "Physical structural damage identified (cracked main branches, bark stripping, or trunk cavity). Safely trim loose or hanging limbs to prevent infection, apply trunk wrap if needed, and evaluate structural stability.",
    'Dead': "No viable foliage or active cambium layer observed. Consult a certified local arborist for immediate safety risk assessment and removal planning to prevent property hazard or insect harborage."
}

def load_or_init_model():
    """Loads existing MobileNetV2 model or reloads when checkpoint updates."""
    global _model, _model_mtime
    
    if os.path.exists(MODEL_PATH):
        current_mtime = os.path.getmtime(MODEL_PATH)
        if _model is not None and current_mtime == _model_mtime:
            return _model
            
        print(f"Loading MobileNetV2 model from {MODEL_PATH}...")
        try:
            _model = keras.models.load_model(MODEL_PATH)
            _model_mtime = current_mtime
            return _model
        except Exception as e:
            print(f"Error loading saved model: {e}. Rebuilding model...")

    if _model is not None:
        return _model

    # If model doesn't exist yet, import builder from model directory and create it
    import sys
    model_dir = os.path.join(PROJECT_ROOT, 'model')
    if model_dir not in sys.path:
        sys.path.append(model_dir)
    
    from init_model import initialize_and_save_model
    _model = initialize_and_save_model(MODEL_PATH)
    return _model

def extract_visual_feature_scores(raw_rgb_np: np.ndarray) -> np.ndarray:
    """
    Computes computer vision domain scores for plant health:
    - Excess Green Index (ExG): Chlorophyll vitality
    - Necrotic Spot Density: Fungal/blight lesions (dark brown/black spots on foliage)
    - Chlorosis Index: Leaf yellowing/decay
    - Dead Biomass / Wood Ratio: Dry leafless biomass
    - Structural Cavity Ratio: Trunk split cracks and inner cavity exposure

    Returns normalized array of scores for ['Damaged', 'Dead', 'Diseased', 'Healthy'].
    """
    R = raw_rgb_np[:, :, 0].astype(np.float32)
    G = raw_rgb_np[:, :, 1].astype(np.float32)
    B = raw_rgb_np[:, :, 2].astype(np.float32)
    total_pixels = raw_rgb_np.shape[0] * raw_rgb_np.shape[1]

    # 1. Green Vegetation & Excess Green Index (ExG)
    green_mask = (G >= R) & (G > B) & (G > 35)
    green_fraction = np.sum(green_mask) / total_pixels

    # 2. Foliage presence & Necrotic Spot Mask (Dark brown/black spots ON foliage: R > G * 1.25)
    necrotic_spot_mask = (
        (R > G * 1.25) & (R > B + 15) & 
        (R < 120) & (G < 95) & (B < 75) & 
        (R > 15)
    )
    necrotic_ratio = np.sum(necrotic_spot_mask) / total_pixels

    # 3. Chlorosis (Yellowing) Leaf Index (R & G are near equal, sickly pale yellow-brown)
    yellow_mask = (R > 125) & (G > 115) & (B < 110) & (abs(R - G) <= 25) & (R > B + 35)
    chlorosis_ratio = np.sum(yellow_mask) / total_pixels

    # 4. Dead Biomass & Bare Wood Ratio
    dead_wood_mask = (
        (abs(R - G) < 25) & (abs(G - B) < 25) & 
        (R < 160) & (G < 160) & (B < 160) & 
        (~green_mask) & (~yellow_mask)
    )
    dead_ratio = np.sum(dead_wood_mask) / total_pixels

    # 5. Damaged Trunk Cavity & Inner Wood Exposure
    cavity_mask = (R < 30) & (G < 20) & (B < 15)
    exposed_inner_wood = (R > 140) & (G > 105) & (B > 65) & (R > G) & (G > B) & (R < 210)
    damaged_ratio = (np.sum(cavity_mask) + np.sum(exposed_inner_wood)) / total_pixels

    v_damaged = 0.01
    v_dead = 0.01
    v_diseased = 0.01
    v_healthy = 0.01

    if green_fraction > 0.15:
        # Image contains active green foliage
        if necrotic_ratio > 0.03 or chlorosis_ratio > 0.06:
            # Active foliage exhibits necrotic spot lesions or chlorotic yellowing decay
            v_diseased += 0.95 + (necrotic_ratio * 4.0) + (chlorosis_ratio * 3.0)
            v_healthy += 0.02
        else:
            # Healthy green foliage without significant disease symptoms
            v_healthy += 0.95 + (green_fraction * 0.4)
            v_diseased += 0.01
    elif chlorosis_ratio > 0.08:
        # Widespread foliage chlorosis / decay
        v_diseased += 0.95 + (chlorosis_ratio * 3.0)
        v_healthy += 0.02
    elif damaged_ratio > 0.04:
        # Trunk structural damage, split cavity, exposed inner wood
        v_damaged += 0.95 + (damaged_ratio * 2.0)
    elif dead_ratio > 0.25:
        # Bare leafless branches, dry withered wood
        v_dead += 0.95 + (dead_ratio * 0.5)
    else:
        if green_fraction > 0.05:
            v_healthy += 0.80
        elif chlorosis_ratio > 0.03 or necrotic_ratio > 0.01:
            v_diseased += 0.80
        elif damaged_ratio > 0.03:
            v_damaged += 0.80
        else:
            v_dead += 0.80

    v_scores = np.array([v_damaged, v_dead, v_diseased, v_healthy], dtype=np.float32)
    return v_scores / np.sum(v_scores)

def predict_tree_health(image_bytes: bytes):
    """
    Takes raw image bytes, runs MobileNetV2 model inference + computer vision feature analysis,
    and returns predicted class, calibrated confidence percentage, and recommendation.
    """
    model = load_or_init_model()
    
    # Load image using Pillow
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_image = image.resize((224, 224))
    
    # Convert image to numpy array & preprocess for MobileNetV2
    raw_img_np = np.array(resized_image, dtype=np.uint8)
    img_array = raw_img_np.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # 1. Deep Learning Softmax Predictions
    nn_predictions = model.predict(img_array, verbose=0)[0] # ['Damaged', 'Dead', 'Diseased', 'Healthy']
    
    # 2. Domain Visual Feature Analysis
    visual_scores = extract_visual_feature_scores(raw_img_np)
    
    # 3. Adaptive Domain Ensemble Blend
    v_max_score = np.max(visual_scores)
    if v_max_score > 0.80:
        w_visual = 0.92
        w_nn = 0.08
    else:
        w_visual = 0.60
        w_nn = 0.40

    combined_scores = (w_nn * nn_predictions) + (w_visual * visual_scores)
    combined_scores = combined_scores / np.sum(combined_scores)
    
    # Calculate top predicted class and confidence score
    max_idx = int(np.argmax(combined_scores))
    confidence_val = float(combined_scores[max_idx])
    
    predicted_class = CLASSES[max_idx]
    confidence_percentage = round(confidence_val * 100, 2)
    
    recommendation = RECOMMENDATIONS.get(
        predicted_class,
        "Perform regular tree health inspections and consult an arborist for detailed diagnosis."
    )
    
    all_predictions = {
        CLASSES[i]: round(float(combined_scores[i]) * 100, 2) 
        for i in range(len(CLASSES))
    }
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence_percentage,
        "recommendation": recommendation,
        "all_predictions": all_predictions
    }


# ========================= 🚀 Imports =========================
import os
import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, GlobalAveragePooling2D, Conv2D, \
    MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal
import albumentations as A
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ========================= 🚀 GPU Configuration =========================
def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Configure GPU memory limit (correct API)
            if len(gpus) > 0:
                # For TensorFlow 2.x, use this approach
                tf.config.experimental.set_memory_growth(gpus[0], True)
                # Set memory limit if needed (optional)
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpus[0],
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)]  # 10GB limit
                # )

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"🚀 {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            print(f"✅ GPU memory growth enabled")

            # Test GPU availability
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                print(f"✅ GPU test successful: {test_tensor.device}")

        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
    else:
        print("❌ No GPU found, using CPU")


setup_gpu()

# ========================= 🚀 Configuration =========================
DATA_ROOT = "/kaggle/input/interr/final_interior"
EXTRACTED_FEATURE_CACHE = "/kaggle/working/feature_cache.npz"
MODEL_DIR = "/kaggle/working/models"
AUGMENTED_DATA_ROOT = "/kaggle/working/augmented_dataset"
LOG_DIR = "/kaggle/working/logs"
AUGMENTATION_VIS_DIR = "/kaggle/working/augmentation_visualization"

EXPECTED_CLASSES = [
    "Poor_(Needs_Attention)",
    "Fair_(Average_Condition)",
    "Good_(Well_Maintained)",
    "luxurious_(Top_Condition)"
]

# ========================= 🚀 OPTIMIZED Class Balancing Configuration =========================
CLASS_AUGMENTATION_FACTORS = {
    "Poor_(Needs_Attention)": 14.0,  # 14x augmentation for Poor class
    "Fair_(Average_Condition)": 7.0,  # 7x augmentation for Fair class (INCREASED)
    "Good_(Well_Maintained)": 5.0,  # 5x augmentation for Good class (INCREASED)
    "luxurious_(Top_Condition)": 10.0  # 10x augmentation for Luxurious class
}

# ========================= 🚀 SMART Augmentation Ratios =========================
CLASS_AUGMENTATION_RATIOS = {
    "Poor_(Needs_Attention)": 0.95,  # 95% augmented, 5% original
    "Fair_(Average_Condition)": 0.8,  # 80% augmented, 20% original
    "Good_(Well_Maintained)": 0.7,  # 70% augmented, 30% original
    "luxurious_(Top_Condition)": 0.9  # 90% augmented, 10% original
}

# Create directories
for dir_path in [MODEL_DIR, AUGMENTED_DATA_ROOT, LOG_DIR, AUGMENTATION_VIS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

NUM_CLASSES = 4
KFOLD_SPLITS = 3
EPOCHS = 80
BATCH_SIZE = 32
TARGET_IMAGES_PER_CLASS = 500
IMG_SIZE = (224, 224)
FEATURE_DIM = 1792  # EfficientNetB4 feature dimension


# ========================= 🚀 Professional Handcrafted Feature Extractor =========================
class InteriorFeatureExtractor:
    """
    Professional handcrafted feature extractor for car interior condition assessment
    Following PakWheels-style inspection criteria
    """

    def _init_(self):
        self.feature_names = []

    def extract_material_quality_features(self, image):
        """Extract luxury material indicators (Leather, Wood, Metal, Fabric ratios)"""
        features = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Leather detection (brown, black, beige tones in smooth textures)
        leather_mask = self._detect_leather_regions(hsv, lab)
        features['leather_ratio'] = np.mean(leather_mask)

        # Wood detection (brown tones with specific texture)
        wood_mask = self._detect_wood_regions(hsv)
        features['wood_ratio'] = np.mean(wood_mask)

        # Metal detection (silver, chrome, metallic reflections)
        metal_mask = self._detect_metal_regions(hsv, lab)
        features['metal_ratio'] = np.mean(metal_mask)

        # Fabric detection (textured, non-smooth areas)
        fabric_mask = self._detect_fabric_regions(image, lab)
        features['fabric_ratio'] = np.mean(fabric_mask)

        # Material quality score (luxury indicator)
        features['luxury_material_score'] = (
                features['leather_ratio'] * 0.4 +
                features['wood_ratio'] * 0.3 +
                features['metal_ratio'] * 0.3
        )

        return features

    def extract_condition_features(self, image):
        """Extract wear & tear condition features (Stains, Cracks, Scratches, etc.)"""
        features = {}
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Dust/Soiling detection (low saturation, medium value areas)
        dust_mask = self._detect_dust_soiling(hsv)
        features['dust_score'] = np.mean(dust_mask)

        # Stains detection (irregular color patches)
        stains_mask = self._detect_stains(image, hsv)
        features['stains_ratio'] = np.mean(stains_mask)

        # Cracks detection (linear dark features in bright areas)
        cracks_mask = self._detect_cracks(gray)
        features['cracks_score'] = np.mean(cracks_mask)

        # Holes detection (dark circular/irregular regions)
        holes_mask = self._detect_holes(gray)
        features['holes_score'] = np.mean(holes_mask)

        # Fading detection (low contrast, washed-out colors)
        fading_score = self._detect_fading(image, hsv)
        features['fading_score'] = fading_score

        # Scratches detection (high-frequency linear features)
        scratches_mask = self._detect_scratches(gray)
        features['scratches_score'] = np.mean(scratches_mask)

        # Seat wrinkles/sagging (texture irregularity)
        wrinkles_score = self._detect_wrinkles(gray)
        features['wrinkles_score'] = wrinkles_score

        # Overall condition score (higher = worse condition)
        features['overall_condition_score'] = (
                features['dust_score'] * 0.2 +
                features['stains_ratio'] * 0.25 +
                features['cracks_score'] * 0.2 +
                features['holes_score'] * 0.15 +
                features['fading_score'] * 0.1 +
                features['scratches_score'] * 0.1
        )

        return features

    def extract_general_appearance_features(self, image):
        """Extract general appearance features (Brightness, Contrast, Color, Texture)"""
        features = {}
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Brightness and contrast
        features['brightness'] = np.mean(gray)
        features['contrast'] = np.std(gray)

        # Color histograms (RGB)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([image], [i], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            for j, val in enumerate(hist):
                features[f'{color}hist{j}'] = val

        # HSV histograms
        for i, channel in enumerate(['h', 's', 'v']):
            hist = cv2.calcHist([hsv], [i], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            for j, val in enumerate(hist):
                features[f'{channel}hist{j}'] = val

        # Texture features using LBP
        texture_features = self._extract_texture_features(gray)
        features.update(texture_features)

        # Colorfulness metric
        features['colorfulness'] = self._calculate_colorfulness(image)

        return features

    def _detect_leather_regions(self, hsv, lab):
        """Detect leather regions based on color and texture"""
        try:
            # Leather colors: browns, blacks, beiges
            lower_brown = np.array([10, 50, 50])
            upper_brown = np.array([20, 200, 200])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

            lower_beige = np.array([15, 30, 150])
            upper_beige = np.array([25, 80, 255])
            beige_mask = cv2.inRange(hsv, lower_beige, upper_beige)

            # Smooth texture (low variance in L channel)
            l_channel = lab[:, :, 0]
            texture_smooth = cv2.GaussianBlur(l_channel, (15, 15), 0)
            texture_var = cv2.Laplacian(texture_smooth, cv2.CV_64F).var()

            # Create smooth mask based on variance threshold
            smooth_mask = np.ones_like(l_channel, dtype=np.uint8)
            if texture_var < 1000:
                smooth_mask = (smooth_mask * 255).astype(np.uint8)
            else:
                smooth_mask = np.zeros_like(l_channel, dtype=np.uint8)

            # Combine masks using proper bitwise operations
            leather_mask = cv2.bitwise_or(brown_mask, beige_mask)
            leather_mask = cv2.bitwise_and(leather_mask, smooth_mask)

            return leather_mask.astype(np.float32) / 255.0

        except Exception:
            return np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.float32)

    def _detect_wood_regions(self, hsv):
        """Detect wood regions based on color patterns"""
        try:
            # Wood colors: various brown tones
            lower_wood1 = np.array([5, 50, 50])
            upper_wood1 = np.array([15, 200, 200])
            wood_mask1 = cv2.inRange(hsv, lower_wood1, upper_wood1)

            lower_wood2 = np.array([15, 30, 30])
            upper_wood2 = np.array([25, 150, 150])
            wood_mask2 = cv2.inRange(hsv, lower_wood2, upper_wood2)

            wood_mask = cv2.bitwise_or(wood_mask1, wood_mask2)
            return wood_mask.astype(np.float32) / 255.0

        except Exception:
            return np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.float32)

    def _detect_metal_regions(self, hsv, lab):
        """Detect metal/chrome regions"""
        try:
            # Metallic colors: grays, silvers with high reflectance
            lower_metal = np.array([0, 0, 150])
            upper_metal = np.array([180, 50, 255])
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)

            # High lightness in LAB space
            l_channel = lab[:, :, 0]
            high_lightness = ((l_channel > 180) * 255).astype(np.uint8)

            metal_mask = cv2.bitwise_and(metal_mask, high_lightness)
            return metal_mask.astype(np.float32) / 255.0

        except Exception:
            return np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.float32)

    def _detect_fabric_regions(self, image, lab):
        """Detect fabric regions based on texture"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Medium texture areas (not too smooth, not too rough)
            if texture > 500 and texture < 5000:
                fabric_mask = np.ones_like(gray, dtype=np.uint8) * 255
            else:
                fabric_mask = np.zeros_like(gray, dtype=np.uint8)

            return fabric_mask.astype(np.float32) / 255.0

        except Exception:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    def _detect_dust_soiling(self, hsv):
        """Detect dust and soiling areas"""
        try:
            # Dust: low saturation, medium value
            lower_dust = np.array([0, 0, 50])
            upper_dust = np.array([180, 50, 150])
            dust_mask = cv2.inRange(hsv, lower_dust, upper_dust)
            return dust_mask.astype(np.float32) / 255.0

        except Exception:
            return np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.float32)

    def _detect_stains(self, image, hsv):
        """Detect stain regions"""
        try:
            saturation = hsv[:, :, 1]
            stain_mask = ((saturation < 50) * 255).astype(np.uint8)
            return stain_mask.astype(np.float32) / 255.0

        except Exception:
            return np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.float32)

    def _detect_cracks(self, gray):
        """Detect crack patterns"""
        try:
            # Cracks: dark linear features
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cracks = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            return cracks.astype(np.float32) / 255.0

        except Exception:
            return np.zeros_like(gray, dtype=np.float32)

    def _detect_holes(self, gray):
        """Detect hole regions"""
        try:
            # Holes: dark circular regions
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
            return thresh.astype(np.float32) / 255.0

        except Exception:
            return np.zeros_like(gray, dtype=np.float32)

    def _detect_fading(self, image, hsv):
        """Detect color fading"""
        try:
            # Fading: low color saturation and contrast
            avg_saturation = np.mean(hsv[:, :, 1])
            contrast = np.std(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
            fading_score = max(0, 1 - (avg_saturation / 100 + contrast / 100) / 2)
            return fading_score

        except Exception:
            return 0.0

    def _detect_scratches(self, gray):
        """Detect scratch patterns"""
        try:
            # Scratches: thin, bright lines on dark background
            edges = cv2.Canny(gray, 100, 200)
            kernel = np.ones((1, 5), np.uint8)
            scratches = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            return scratches.astype(np.float32) / 255.0

        except Exception:
            return np.zeros_like(gray, dtype=np.float32)

    def _detect_wrinkles(self, gray):
        """Detect wrinkles and sagging in seats"""
        try:
            # Wrinkles: high frequency texture variations
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()
            wrinkles_score = min(1.0, texture / 5000)  # Normalize
            return wrinkles_score

        except Exception:
            return 0.0

    def _extract_texture_features(self, gray):
        """Extract LBP texture features"""
        features = {}

        try:
            # Simple texture metrics
            features['texture_variance'] = np.var(gray)
            features['texture_entropy'] = self._calculate_entropy(gray)

            # Simplified LBP features
            lbp_features = self._simplified_lbp(gray)
            for i, val in enumerate(lbp_features):
                features[f'lbp_hist_{i}'] = val

        except Exception:
            features['texture_variance'] = 0.0
            features['texture_entropy'] = 0.0
            for i in range(8):
                features[f'lbp_hist_{i}'] = 0.0

        return features

    def _simplified_lbp(self, image):
        """Simplified LBP implementation for texture analysis"""
        try:
            height, width = image.shape
            lbp = np.zeros_like(image)

            # Simple 3x3 neighborhood LBP
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    center = image[i, j]
                    binary_pattern = 0
                    positions = [
                        (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                        (i, j - 1), (i, j + 1),
                        (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
                    ]

                    for idx, (x, y) in enumerate(positions):
                        if image[x, y] >= center:
                            binary_pattern |= (1 << idx)

                    lbp[i, j] = binary_pattern

            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=8, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-8

            return hist

        except Exception:
            return np.zeros(8, dtype=np.float32)

    def _calculate_entropy(self, image):
        """Calculate image entropy"""
        try:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.astype(np.float32) + 1e-8
            hist /= hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            return entropy
        except Exception:
            return 0.0

    def _calculate_colorfulness(self, image):
        """Calculate colorfulness metric"""
        try:
            # Split image into RGB channels
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            # Calculate rg and yb
            rg = np.abs(r - g)
            yb = np.abs(0.5 * (r + g) - b)

            # Calculate mean and standard deviation
            rg_mean, rg_std = np.mean(rg), np.std(rg)
            yb_mean, yb_std = np.mean(yb), np.std(yb)

            # Calculate colorfulness
            colorfulness = np.sqrt(rg_std * 2 + yb_std2) + 0.3 * np.sqrt(rg_mean2 + yb_mean * 2)
            return colorfulness / 100.0  # Normalize
        except Exception:
            return 0.0

    def extract_all_features(self, image_path):
        """Extract all handcrafted features from an image"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMG_SIZE)

            # Extract all feature groups
            material_features = self.extract_material_quality_features(image)
            condition_features = self.extract_condition_features(image)
            appearance_features = self.extract_general_appearance_features(image)

            # Combine all features
            all_features = {}
            all_features.update(material_features)
            all_features.update(condition_features)
            all_features.update(appearance_features)

            feature_vector = np.array(list(all_features.values()))

            # Validate feature vector
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return None

            return feature_vector

        except Exception:
            return None


# ========================= 🚀 Enhanced Feature Extraction with Handcrafted Features =========================
class EnhancedFeatureExtractor:
    """Handles both CNN and handcrafted feature extraction"""

    def _init_(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.handcrafted_extractor = InteriorFeatureExtractor()

        print("🚀 Loading EfficientNetB4 on GPU...")
        # Force GPU usage for the backbone
        with tf.device('/GPU:0'):
            self.backbone = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(*img_size, 3)
            )
            self.backbone.trainable = False
        print("✅ EfficientNetB4 loaded and frozen on GPU")

    def extract_combined_features(self, image_path, augmentor=None,
                                  is_training=False, augmentation_round=0, class_name=None):
        """Extract both CNN and handcrafted features"""
        try:
            # Extract CNN features
            cnn_features = self._extract_cnn_features(image_path, augmentor, is_training, augmentation_round,
                                                      class_name)

            # Extract handcrafted features
            handcrafted_features = self.handcrafted_extractor.extract_all_features(image_path)

            if cnn_features is not None and handcrafted_features is not None:
                # Combine features into single feature vector
                combined_features = np.concatenate([cnn_features, handcrafted_features])
                return combined_features, len(handcrafted_features)
            elif cnn_features is not None:
                # Fallback to CNN features only
                return cnn_features, 0
            else:
                return None, 0

        except Exception:
            return None, 0

    def _extract_cnn_features(self, image_path, augmentor=None,
                              is_training=False, augmentation_round=0, class_name=None):
        """Extract CNN features with class-balanced augmentation - GPU OPTIMIZED"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            # Apply class-balanced augmentation if training AND augmentor is provided
            was_augmented = False
            if augmentor and is_training and class_name:
                img, was_augmented = augmentor.augment_image(
                    img, is_training=True, augmentation_round=augmentation_round, class_name=class_name
                )

            # Preprocess for EfficientNet
            img_processed = eff_preprocess(img.astype(np.float32))
            img_batch = np.expand_dims(img_processed, axis=0)

            # Force GPU prediction
            with tf.device('/GPU:0'):
                features = self.backbone.predict(img_batch, verbose=0)
            return features.flatten()

        except Exception:
            return None

    def extract_car_features(self, car_image_paths, car_class, augmentor=None,
                             is_training=False, epoch=0):
        """Extract features for car with combined approach - GPU OPTIMIZED"""
        car_features = []
        handcrafted_dims = 0
        augmentation_stats = {"augmented": 0, "total": 0}
        successful_extractions = 0

        for idx, img_path in enumerate(car_image_paths):
            try:
                augmentation_round = epoch * len(car_image_paths) + idx
                features, hc_dim = self.extract_combined_features(
                    img_path, augmentor, is_training, augmentation_round, car_class
                )

                if features is not None:
                    car_features.append(features)
                    handcrafted_dims = hc_dim
                    augmentation_stats["total"] += 1
                    successful_extractions += 1

                    # Track augmentation if training and augmentor provided
                    if augmentor and is_training:
                        # Check if this image should have been augmented based on class
                        if augmentor.should_augment(augmentation_round, car_class):
                            augmentation_stats["augmented"] += 1

            except Exception:
                continue

        return np.array(
            car_features) if car_features else None, augmentation_stats, successful_extractions, handcrafted_dims

    def aggregate_features(self, car_features, method='mean'):
        """Aggregate per-image features to car-level"""
        if car_features is None or len(car_features) == 0:
            return None

        if method == 'mean':
            return np.mean(car_features, axis=0)
        elif method == 'max':
            return np.max(car_features, axis=0)
        elif method == 'std_mean':
            feature_mean = np.mean(car_features, axis=0)
            feature_std = np.std(car_features, axis=0)
            return np.concatenate([feature_mean, feature_std])
        else:
            return np.mean(car_features, axis=0)


# ========================= 🚀 ENHANCED SMART Augmentation Strategy =========================
class ProfessionalAugmentor:
    """
    ENHANCED: SMART professional augmentation strategy with dual balancing:
    - OPTIMIZED Class Augmentation Factors (14x, 7x, 5x, 10x)
    - SMART Class-wise Augmentation Ratios (95%, 80%, 70%, 90%)
    - All classes get enhanced augmentation
    - Poor class gets maximum boost (95% augmented + 14x factor)
    - Realistic transforms preserving car interior recognition
    """

    def __init__(self, img_size=IMG_SIZE,
                 class_augmentation_factors=None,
                 class_augmentation_ratios=None):
        self.img_size = img_size
        self.class_augmentation_factors = class_augmentation_factors or CLASS_AUGMENTATION_FACTORS
        self.class_augmentation_ratios = class_augmentation_ratios or CLASS_AUGMENTATION_RATIOS
        self.train_transform = self._create_train_augmentation()
        self.val_transform = self._create_val_transform()

        print("🎯 ENHANCED SMART Augmentation Configuration:")
        print("📊 OPTIMIZED Class Augmentation Factors:")
        for class_name, factor in self.class_augmentation_factors.items():
            print(f"   {class_name}: {factor}x augmentation")

        print("📊 SMART Class-wise Augmentation Ratios:")
        for class_name, ratio in self.class_augmentation_ratios.items():
            original_percent = (1 - ratio) * 100
            print(f"   {class_name}: {ratio * 100:.0f}% augmented, {original_percent:.0f}% original")

    def _create_train_augmentation(self):
        """Create realistic augmentation pipeline for car interiors"""
        return A.Compose([
            # Geometric transforms - small and realistic
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_REFLECT),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5,
                border_mode=cv2.BORDER_REFLECT
            ),

            # Photometric transforms - realistic lighting variations
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

            # Noise and quality variations
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),

            # Occlusion and dropout - realistic for interiors
            A.CoarseDropout(
                max_holes=6, max_height=16, max_width=16,
                min_holes=1, min_height=8, min_width=8, p=0.3
            ),

        ], p=1.0)

    def _create_val_transform(self):
        """No augmentation for validation/test"""
        return A.Compose([])

    def should_augment(self, augmentation_round, class_name):
        """Determine if this specific image should be augmented based on SMART class-wise ratios"""
        if class_name not in self.class_augmentation_ratios:
            return False

        # Use SMART class-wise augmentation ratio
        class_ratio = self.class_augmentation_ratios[class_name]

        # Use deterministic augmentation based on round and class
        class_hash = hash(class_name) % 1000
        deterministic_value = (augmentation_round + class_hash) % 100
        return deterministic_value < (class_ratio * 100)

    def get_augmentation_stats(self, car_labels, car_image_paths, num_epochs=1):
        """Calculate expected augmentation statistics for monitoring with SMART ratios"""
        print("\n📊 ENHANCED AUGMENTATION STATISTICS:")
        print("=" * 60)

        total_original = 0
        total_augmented = 0
        class_stats = {}

        for class_name in self.class_augmentation_ratios.keys():
            class_cars = [car_id for car_id, label in car_labels.items() if label == class_name]
            class_images = sum(len(car_image_paths[car_id]) for car_id in class_cars if car_id in car_image_paths)

            # Get SMART ratio for this class
            augmentation_ratio = self.class_augmentation_ratios[class_name]
            augmentation_factor = self.class_augmentation_factors[class_name]

            # Calculate expected augmented and original images
            expected_augmented = int(class_images * num_epochs * augmentation_ratio * (augmentation_factor / 10))
            expected_original = int(class_images * num_epochs * (1 - augmentation_ratio))

            class_stats[class_name] = {
                'original_images': class_images,
                'expected_augmented': expected_augmented,
                'expected_original': expected_original,
                'augmentation_factor': augmentation_factor,
                'augmentation_ratio': augmentation_ratio
            }

            total_original += expected_original
            total_augmented += expected_augmented

            print(f"{class_name:<25}: {augmentation_factor:>2}x + {augmentation_ratio * 100:>2.0f}% → "
                  f"{expected_augmented:>5} aug + {expected_original:>4} orig")

        total = total_original + total_augmented
        if total > 0:
            aug_percentage = (total_augmented / total) * 100
            print(f"\n📈 Overall Statistics:")
            print(f"   Total Augmented: {total_augmented}")
            print(f"   Total Original: {total_original}")
            print(f"   Grand Total: {total}")
            print(f"   Effective Augmentation Rate: {aug_percentage:.1f}%")
        print("=" * 60)

        return class_stats

    def augment_image(self, image, is_training=True, augmentation_round=0, class_name=None):
        """
        Apply SMART class-balanced augmentation with dual factors
        Returns augmented image and augmentation flag
        """
        if is_training and class_name and self.should_augment(augmentation_round, class_name):
            try:
                augmented = self.train_transform(image=image)
                augmented_image = augmented['image']

                if augmented_image is None or augmented_image.size == 0:
                    return image, False

                augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
                return augmented_image, True

            except Exception:
                return image, False
        else:
            return image, False


# ========================= 🚀 Data Structure =========================
class CarDataset:
    """Manages car-level dataset with proper grouping"""

    def __init__(self, data_root, expected_classes):
        self.data_root = data_root
        self.expected_classes = expected_classes
        self.car_data = defaultdict(list)
        self.car_labels = {}

    def discover_cars(self):
        """Discover all cars and their images - FIXED VERSION"""
        print("🔍 Discovering cars and images...")

        for class_name in self.expected_classes:
            class_path = os.path.join(self.data_root, class_name)
            print(f"Checking: {class_path}")

            if not os.path.exists(class_path):
                print(f"⚠ Class directory not found: {class_path}")
                continue

            # Get all car folders (subdirectories)
            car_folders = [f for f in os.listdir(class_path)
                           if os.path.isdir(os.path.join(class_path, f))]

            print(f"Found {len(car_folders)} car folders in {class_name}")

            for car_folder in car_folders:
                car_path = os.path.join(class_path, car_folder)
                car_id = f"{class_name}_{car_folder}"

                # Look for images in each car folder
                image_paths = []
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    pattern = os.path.join(car_path, f"*{ext}")
                    found = glob(pattern)
                    image_paths.extend(found)

                if image_paths:
                    self.car_data[car_id] = image_paths
                    self.car_labels[car_id] = class_name
                    print(f"✅ Found {len(image_paths)} images in {car_folder}")

        total_images = sum(len(imgs) for imgs in self.car_data.values())
        print(f"✅ Discovered {len(self.car_data)} cars with {total_images} total images")
        return self.car_data, self.car_labels

    def get_class_distribution(self):
        """Get car count per class"""
        class_counts = defaultdict(int)
        for class_name in self.car_labels.values():
            class_counts[class_name] += 1
        return class_counts


# ========================= 🚀 Model Factory =========================
class ModelFactory:
    """Factory for creating model architectures"""

    @staticmethod
    def create_neural_classifier(input_dim, num_classes, dropout_rate=0.3):
        """Create neural network classifier - GPU OPTIMIZED"""
        with tf.device('/GPU:0'):
            model = Sequential([
                Dense(512, activation='relu', input_shape=(input_dim,),
                      kernel_initializer=HeNormal()),
                BatchNormalization(),
                Dropout(dropout_rate),

                Dense(256, activation='relu', kernel_initializer=HeNormal()),
                BatchNormalization(),
                Dropout(dropout_rate),

                Dense(128, activation='relu', kernel_initializer=HeNormal()),
                BatchNormalization(),
                Dropout(dropout_rate / 2),

                Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        return model


# ========================= 🚀 Enhanced Professional Training Pipeline =========================
# ========================= 🚀 Enhanced Professional Training Pipeline =========================
class ProfessionalTrainingPipeline:
    """Handles complete training with SMART class-balanced augmentation and professional logging"""

    def __init__(self, feature_extractor, augmentor):
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.epoch_stats = []

    def prepare_features(self, car_dataset, use_cache=True):
        """Prepare features with professional caching - GPU OPTIMIZED"""
        cache_file = EXTRACTED_FEATURE_CACHE

        if use_cache and os.path.exists(cache_file):
            print("📁 Loading cached features...")
            try:
                cache_data = np.load(cache_file, allow_pickle=True)
                X = cache_data['X']
                y = cache_data['y']
                car_ids = cache_data['car_ids']
                car_image_paths = cache_data['car_image_paths'].item()

                # Try to load car_labels, but handle if it doesn't exist
                if 'car_labels' in cache_data:
                    car_labels = cache_data['car_labels'].item()
                else:
                    print("⚠ car_labels not found in cache, reconstructing from data...")
                    car_labels = car_dataset.car_labels

                print(f"✅ Loaded {X.shape[0]} cars from cache")
                return X, y, car_ids, car_image_paths, car_labels

            except Exception as e:
                print(f"⚠ Error loading cache: {e}, extracting features from scratch...")
                use_cache = False

        if not use_cache or not os.path.exists(cache_file):
            print("🔧 Extracting features from scratch...")
            print("💡 Using Combined Features: CNN + Handcrafted (Material + Condition + Appearance)")
            print("🎯 GPU Acceleration: Enabled for feature extraction")
            car_data, car_labels = car_dataset.car_data, car_dataset.car_labels
            car_ids = list(car_data.keys())

            X_list, y_list, valid_car_ids, car_image_paths_dict = [], [], [], {}
            total_successful = 0
            total_processed = 0

            # Use tqdm for clean progress bar
            for car_id in tqdm(car_ids, desc="Extracting combined features"):
                image_paths = car_data[car_id]
                car_class = car_labels[car_id]
                car_features, _, successful, handcrafted_dims = self.feature_extractor.extract_car_features(
                    image_paths, car_class, is_training=False  # No augmentation for base features
                )

                if car_features is not None and len(car_features) > 0:
                    aggregated_features = self.feature_extractor.aggregate_features(
                        car_features, method='mean'
                    )
                    if aggregated_features is not None:
                        X_list.append(aggregated_features)
                        y_list.append(car_class)
                        valid_car_ids.append(car_id)
                        car_image_paths_dict[car_id] = image_paths
                        total_successful += successful
                        total_processed += len(image_paths)

            X = np.array(X_list)
            y = np.array(y_list)
            car_ids = np.array(valid_car_ids)

            # Final summary only
            print(f"✅ All features extracted successfully")
            print(f"📊 Final Feature Dimensions - CNN: {FEATURE_DIM}, Handcrafted: {handcrafted_dims}")
            print(f"📊 Total Cars Processed: {len(X_list)}")
            print(
                f"📊 Images Successfully Processed: {total_successful}/{total_processed} ({total_successful / total_processed * 100:.1f}%)")

            # Save with all necessary keys including car_labels
            np.savez(cache_file, X=X, y=y, car_ids=car_ids,
                     car_image_paths=car_image_paths_dict, car_labels=car_labels)
            car_image_paths = car_image_paths_dict

        return X, y, car_ids, car_image_paths, car_labels

    def perform_stratified_split(self, X, y, car_ids, test_size=0.2, random_state=42):
        """Perform stratified train-test split at car level"""
        y_encoded = self.label_encoder.fit_transform(y)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        groups = np.arange(len(car_ids))

        train_idx, test_idx = next(sgkf.split(X, y_encoded, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        car_ids_train, car_ids_test = car_ids[train_idx], car_ids[test_idx]

        print(f"✅ Dataset split - Train: {X_train.shape[0]} cars, Test: {X_test.shape[0]} cars")
        return X_train, X_test, y_train, y_test, car_ids_train, car_ids_test

    def create_professional_callbacks(self):
        """Create professional callbacks for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        callbacks = [
            # Save best model
            ModelCheckpoint(
                os.path.join(MODEL_DIR, f"best_model_{timestamp}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            # Save latest model
            ModelCheckpoint(
                os.path.join(MODEL_DIR, "latest_model.h5"),
                save_best_only=False,
                save_weights_only=False,
                verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(LOG_DIR, timestamp),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]

        return callbacks

    def train_with_professional_logging(self, X_train, y_train, X_test, y_test,
                                        car_ids_train, car_image_paths, car_labels):
        """Train model with professional epoch-level logging and SMART augmentation - GPU OPTIMIZED"""
        print("\n🎯 Starting Professional Training with SMART Augmentation")
        print("🎯 GPU Acceleration: Enabled for model training")
        print("🎯 ENHANCED SMART Augmentation Strategy:")
        print("📊 OPTIMIZED Class Augmentation Factors:")
        for class_name, factor in CLASS_AUGMENTATION_FACTORS.items():
            print(f"   {class_name}: {factor}x augmentation")

        print("📊 SMART Class-wise Augmentation Ratios:")
        for class_name, ratio in CLASS_AUGMENTATION_RATIOS.items():
            original_percent = (1 - ratio) * 100
            print(f"   {class_name}: {ratio * 100:.0f}% augmented, {original_percent:.0f}% original")
        print("=" * 60)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create model on GPU
        print("🔄 Creating neural network classifier on GPU...")
        model = ModelFactory.create_neural_classifier(
            X_train_scaled.shape[1], len(self.label_encoder.classes_)
        )

        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Create callbacks
        callbacks = self.create_professional_callbacks()

        print("\n📊 Starting training with real-time progress...")
        print("✅ Epoch progress bars will be visible")
        print("✅ Model will be saved automatically")
        print("✅ Learning rate adjustments will be shown")
        print("=" * 50)

        # Train the model with proper progress display
        history = model.fit(
            X_train_scaled, y_train_encoded,
            validation_data=(X_test_scaled, y_test_encoded),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,  # THIS IS CRUCIAL - shows progress bars!
            shuffle=True
        )

        # Save final model
        final_model_path = os.path.join(MODEL_DIR, "professional_interior_classifier_final.h5")
        model.save(final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")

        return model, history.history['accuracy'], history.history['val_accuracy']


# ========================= 🚀 Model Factory =========================
class ModelFactory:
    """Factory for creating model architectures"""

    @staticmethod
    def create_neural_classifier(input_dim, num_classes, dropout_rate=0.3):
        """Create neural network classifier - GPU OPTIMIZED"""
        with tf.device('/GPU:0'):
            model = Sequential([
                Dense(512, activation='relu', input_shape=(input_dim,),
                      kernel_initializer=HeNormal()),
                BatchNormalization(),
                Dropout(dropout_rate),

                Dense(256, activation='relu', kernel_initializer=HeNormal()),
                BatchNormalization(),
                Dropout(dropout_rate),

                Dense(128, activation='relu', kernel_initializer=HeNormal()),
                BatchNormalization(),
                Dropout(dropout_rate / 2),

                Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', 'sparse_categorical_crossentropy']  # Add more metrics
            )

        print(f"✅ Model created with input_dim: {input_dim}, num_classes: {num_classes}")
        model.summary()
        return model


# ========================= 🚀 Evaluation =========================
class ProfessionalEvaluator:
    """Handles professional model evaluation"""

    @staticmethod
    def evaluate_model(model, scaler, label_encoder, X_test, y_test):
        """Comprehensive model evaluation"""
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred,
                                             target_names=label_encoder.classes_,
                                             digits=4)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"\n🎯 Final Test Accuracy: {accuracy:.4f}")
        print(f"\n📊 Classification Report:\n{class_report}")

        return accuracy, class_report, conf_matrix


# ========================= 🚀 Main Execution =========================
# ========================= 🚀 Main Execution =========================
def main():
    print("🚀 Starting Professional Interior Classification Pipeline")
    print("=" * 60)
    # ... (your existing print statements)

    # Step 1: Setup and data discovery
    print("\n1. 📁 Setting up dataset...")
    car_dataset = CarDataset(DATA_ROOT, EXPECTED_CLASSES)
    car_data, car_labels = car_dataset.discover_cars()

    class_distribution = car_dataset.get_class_distribution()
    print(f"📊 Original Class Distribution: {dict(class_distribution)}")

    if len(car_data) == 0:
        print("❌ No cars found! Please check your data directory.")
        return

    # Step 2: Initialize professional components
    print("\n2. 🔧 Initializing professional components...")
    feature_extractor = EnhancedFeatureExtractor()
    augmentor = ProfessionalAugmentor(
        class_augmentation_factors=CLASS_AUGMENTATION_FACTORS,
        class_augmentation_ratios=CLASS_AUGMENTATION_RATIOS
    )

    # Show expected augmentation statistics
    augmentor.get_augmentation_stats(car_labels, car_data, num_epochs=EPOCHS)

    training_pipeline = ProfessionalTrainingPipeline(feature_extractor, augmentor)
    evaluator = ProfessionalEvaluator()

    # Step 3: Feature extraction
    print("\n3. 🔍 Extracting and aggregating features...")
    # Debug the prepare_features method
    print("Debug: Calling prepare_features...")
    result = training_pipeline.prepare_features(car_dataset, use_cache=True)
    print(f"Result type: {type(result)}")
    if result is None:
        print("prepare_features returned None!")
    return
    else:
    print(f"Result length: {len(result)}")
    X, y, car_ids, car_image_paths, all_car_labels = result

    # Step 4: Data splitting
    print("\n4. 📊 Performing stratified split...")
    X_train, X_test, y_train, y_test, car_ids_train, car_ids_test = \
        training_pipeline.perform_stratified_split(X, y, car_ids)

    # Step 5: Professional training with proper progress display
    print("\n5. 🎯 Starting Professional Training")
    print("   ✅ Epoch progress will be visible")
    print("   ✅ Model will auto-save")
    print("   ✅ Real-time metrics displayed")
    print("=" * 50)

    final_model, train_accuracies, val_accuracies = \
        training_pipeline.train_with_professional_logging(
            X_train, y_train, X_test, y_test, car_ids_train, car_image_paths, all_car_labels
        )

    # Step 6: Professional evaluation
    print("\n6. 📈 Professional Evaluation")
    print("-" * 40)
    accuracy, class_report, conf_matrix = evaluator.evaluate_model(
        final_model, training_pipeline.scaler, training_pipeline.label_encoder,
        X_test, y_test
    )

    # Step 7: Plot training history
    print("\n7. 📊 Generating training plots...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.title('Training History', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss History', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'complete_training_history.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("🎯 PROFESSIONAL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"✅ Final Test Accuracy: {accuracy:.4f}")
    print(f"✅ Models saved to: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":  # Fixed this line - was "_main_" before
    main()
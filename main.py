
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import joblib
import io
import numpy as np
import pandas as pd

app = FastAPI(
    title="Skin Cancer Classification API",
    description=(
        "ResNet-50 feature extraction, ADASYN scaler, "
        "PCA transformation, and SVM classification."
    ),
    version="2.0.0",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: Verify these labels against the original training labels.
CLASS_NAMES = {
    0: "benign",
    1: "malignant",
}
CLASS_MAPPING_VERIFIED = False

# Load pretrained ResNet-50.
weights = ResNet50_Weights.IMAGENET1K_V2
resnet = resnet50(weights=weights)

# Keep convolutional feature maps for global average pooling.
spatial_extractor = nn.Sequential(
    *list(resnet.children())[:-2]
).to(device)
spatial_extractor.eval()

preprocess = weights.transforms()

# Load the ADASYN-trained classification pipeline.
scaler = joblib.load("ada_scaler.pkl")
pca = joblib.load("ada_pca.pkl")
svm = joblib.load("ada_svc.pkl")

# Access the underlying linear SVM used by the calibrated classifier.
base_svc = svm.calibrated_classifiers_[0].estimator

if not hasattr(base_svc, "coef_"):
    raise RuntimeError("The underlying SVM has no linear coefficients.")

if not hasattr(base_svc, "decision_function"):
    raise RuntimeError("The underlying SVM has no decision function.")

if not hasattr(pca, "components_") or not hasattr(scaler, "scale_"):
    raise RuntimeError("PCA or scaler is missing required fitted attributes.")

# Map the SVM weights backwards through PCA and the scaler.
# SVM weights: 446 components.
# PCA projection: 446 components back to 2048 scaled features.
# Scaler adjustment: 2048 scaled features back to original feature scale.
svm_weights = np.asarray(base_svc.coef_[0], dtype=np.float64)

if svm_weights.shape[0] != pca.n_components_:
    raise RuntimeError("SVM coefficient count does not match PCA components.")

lfam_scaled_weights = svm_weights @ pca.components_

lfam_weights = lfam_scaled_weights / np.asarray(
    scaler.scale_, dtype=np.float64
)

if lfam_weights.shape[0] != 2048:
    raise RuntimeError("LFAM weights do not match the 2048 ResNet channels.")
# Validate pipeline dimensions.
if scaler.n_features_in_ != 2048:
    raise RuntimeError(
        f"Scaler expects {scaler.n_features_in_} features, not 2048."
    )

if pca.n_features_in_ != scaler.n_features_in_:
    raise RuntimeError(
        "PCA input dimensions do not match scaler input dimensions."
    )

if pca.n_components_ != 446:
    raise RuntimeError(
        f"Expected 446 PCA components, got {pca.n_components_}."
    )

if getattr(svm, "n_features_in_", None) != pca.n_components_:
    raise RuntimeError(
        "SVM input dimensions do not match PCA output dimensions."
    )

if not hasattr(svm, "predict_proba"):
    raise RuntimeError("The SVM does not support predict_proba.")


def prepare_features(values, estimator):
    """Preserve feature names when required by a fitted estimator."""
    columns = getattr(estimator, "feature_names_in_", None)
    if columns is not None:
        return pd.DataFrame(values, columns=columns)
    return values


@app.get("/")
def root():
    return {
        "message": "Skin Cancer Classification API is running",
        "device": str(device),
        "model_loaded": True,
        "pipeline": (
            "ResNet-50 (2048 features) -> ADASYN scaler "
            "-> PCA (446 components) -> ADASYN SVM"
        ),
        "feature_count_before_pca": int(scaler.n_features_in_),
        "feature_count_after_pca": int(pca.n_components_),
        "pca_used": True,
        "class_mapping": CLASS_NAMES,
        "class_mapping_verified": CLASS_MAPPING_VERIFIED,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    original_width, original_height = image.size

    # Apply the pretrained ResNet-50 preprocessing.
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Extract feature maps: (1, 2048, 7, 7).
   with torch.inference_mode():
    spatial_tensor = spatial_extractor(image_tensor)
    features = (
        spatial_tensor.mean(dim=(2, 3))
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Global average pooling: (2048, 7, 7) -> (1, 2048).
    features = spatial_features.mean(
        axis=(1, 2)
    ).reshape(1, -1)

    if features.shape[1] != scaler.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail=(
                f"ResNet produced {features.shape[1]} features; "
                f"the scaler expects {scaler.n_features_in_}."
            ),
        )

    # Step 1: Standardization.
    scaler_input = prepare_features(features, scaler)
    scaled_features = scaler.transform(scaler_input)

    # Step 2: PCA dimensionality reduction.
    pca_input = prepare_features(scaled_features, pca)
    reduced_features = pca.transform(pca_input)

    # Step 3: SVM classification.
    svm_input = prepare_features(reduced_features, svm)
    predicted_class = int(svm.predict(svm_input)[0])
    probabilities = svm.predict_proba(svm_input)[0]


    # Raw linear SVM decision score.
    decision_score = float(
        base_svc.decision_function(svm_input)[0]
    )

    # Create the LFAM heatmap for the predicted class.
    target_weights = lfam_weights.copy()

    if predicted_class == 0:
        target_weights = -target_weights

    heatmap = np.einsum(
        "c,chw->hw",
        target_weights,
        spatial_features
    )

    # Retain positive spatial evidence and normalize it.
    heatmap = np.maximum(heatmap, 0)
    heatmap_max = float(heatmap.max())

    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_data = heatmap.astype(np.float32).tolist()


    class_probabilities = {
        str(int(label)): float(probability)
        for label, probability in zip(
            svm.classes_, probabilities
        )
    }

    named_probabilities = {
        CLASS_NAMES.get(int(label), f"class_{int(label)}"): float(probability)
        for label, probability in zip(
            svm.classes_, probabilities
        )
    }

    # Shannon entropy measures the spread of predicted probabilities.
    # It is not a clinically validated certainty score.
    epsilon = 1e-12
    safe_probabilities = np.clip(
        probabilities, epsilon, 1.0
    )

    entropy = float(
        -np.sum(safe_probabilities * np.log(safe_probabilities))
    )

    max_entropy = float(np.log(len(probabilities)))

    probability_concentration = (
        float(
            np.clip(
                (1 - entropy / max_entropy) * 100,
                0,
                100,
            )
        )
        if max_entropy > 0
        else 100.0
    )

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES.get(
            predicted_class, "unknown"
        ),
        "class_probabilities": class_probabilities,
        "named_class_probabilities": named_probabilities,
        "feature_count": int(features.shape[1]),
        "pca_components": int(reduced_features.shape[1]),
        "pca_used": True,
        "shannon_entropy": entropy,
        
        "probability_concentration_score_percent": (
            probability_concentration
        ),
        "decision_score": decision_score,
        "decision_score_direction": (
            "toward_malignant_class_1"
            if decision_score > 0
            else "toward_benign_class_0"
            if decision_score < 0
            else "decision_boundary"
        ),
        "interpretability": {
            "available": True,
            "method": "LFAM",
            "target_class": predicted_class,
            "heatmap_data": heatmap_data,
            "heatmap_height": int(heatmap.shape[0]),
            "heatmap_width": int(heatmap.shape[1]),
            "normalization": "0_to_1",
            "overlay_note": (
                "The heatmap is 7x7 and corresponds to the ResNet-50 "
                "224x224 model input. Align the overlay with the same "
                "preprocessed image crop."
            ),
            "validation_notice": (
                "Experimental feature-attribution map; not clinically "
                "validated and not proof of lesion localization."
            ),
        },
        "image": {
            "original_width": int(original_width),
            "original_height": int(original_height),
        },
        "class_mapping_verified": CLASS_MAPPING_VERIFIED,
        "research_notice": (
            "This application is an experimental machine-learning prototype developed for academic research"
            " and educational purposes. It is not a medical diagnostic tool and must not be used to make clinical decisions"
            ". Predictions and estimated probabilities may be inaccurate and have not been validated for clinical use."
            " All results should be interpreted within the context of the research study and must not replace "
            "professional medical assessment or advice from a qualified healthcare professional."
        ),
    }




import sys
import uuid
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from backend.services.bird_details_service import bird_details_service
# ==========================================
# 1. PROJECT PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ==========================================
# 2. IMPORT SERVICES
# ==========================================

from data_transforms import eval_transform

from backend.services.model_service import BirdVisionModel
from backend.services.gradcam_service import GradCAMService

from backend.database import (
    initialize_database,
    save_prediction,
    get_prediction_history,
    delete_prediction,
    HISTORY_IMAGES_DIR
)

# ==========================================
# 3. FASTAPI CONFIGURATION
# ==========================================

app = FastAPI(
    title="BirdVision AI API",
    description=(
        "Bird species identification using ResNet-50 "
        "with Grad-CAM explainability and prediction history."
    ),
    version="3.1.0"
)

# ==========================================
# 4. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. STATIC IMAGE STORAGE
# ==========================================

HISTORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/history-images",
    StaticFiles(directory=str(HISTORY_IMAGES_DIR)),
    name="history-images"
)

# ==========================================
# 6. INITIALIZE DATABASE
# ==========================================

initialize_database()

# ==========================================
# 7. LOAD MODEL AND GRAD-CAM SERVICE
# ==========================================

print("Initializing BirdVision AI...", flush=True)

bird_model = BirdVisionModel()

gradcam_service = GradCAMService(
    bird_model.model
)

print("BirdVision AI initialized successfully!", flush=True)
print("Grad-CAM service initialized!", flush=True)

# ==========================================
# 8. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to BirdVision AI API",
        "status": "running",
        "model": "ResNet-50",
        "species_count": len(bird_model.classes),
        "gradcam_enabled": True,
        "prediction_history_enabled": True,
        "image_storage_enabled": True
    }

# ==========================================
# 9. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": bird_model.model is not None,
        "species_count": len(bird_model.classes),
        "gradcam_enabled": gradcam_service is not None,
        "database_enabled": True,
        "image_storage_enabled": True
    }

# ==========================================
# 10. BIRD PREDICTION + GRAD-CAM + HISTORY
# ==========================================

@app.post("/api/predict")
async def predict_bird(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    saved_image_path = None

    try:

        # --------------------------------------
        # A. READ AND VALIDATE IMAGE
        # --------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # --------------------------------------
        # B. NORMAL PREDICTION
        # --------------------------------------

        prediction_result = bird_model.predict(image)

        # --------------------------------------
        # C. PREPARE IMAGE FOR GRAD-CAM
        # --------------------------------------

        image_tensor = eval_transform(image)

        image_tensor = image_tensor.unsqueeze(0).to(
            bird_model.device
        )

        # --------------------------------------
        # D. GENERATE GRAD-CAM
        # --------------------------------------

        heatmap, predicted_index, confidence = (
            gradcam_service.generate(image_tensor)
        )

        predicted_species = bird_model.classes[
            predicted_index
        ]

        # --------------------------------------
        # E. CREATE VISUALIZATIONS
        # --------------------------------------

        visualizations = (
            gradcam_service.create_visualizations(
                image,
                heatmap
            )
        )

        # --------------------------------------
        # F. SAVE ORIGINAL IMAGE
        # --------------------------------------

        image_extension = (
            ".png" if file.content_type == "image/png"
            else ".jpg"
        )

        unique_filename = (
            f"{uuid.uuid4().hex}{image_extension}"
        )

        saved_image_path = HISTORY_IMAGES_DIR / unique_filename

        # Store a normalized RGB copy of the uploaded image.
        image.save(
            saved_image_path,
            format="PNG" if image_extension == ".png" else "JPEG",
            quality=95
        )

        relative_image_path = (
            saved_image_path.relative_to(PROJECT_ROOT).as_posix()
        )

        # --------------------------------------
        # G. SAVE PREDICTION TO SQLITE
        # --------------------------------------

        try:

            prediction_id = save_prediction(
                filename=file.filename or "uploaded_image",
                predicted_species=prediction_result[
                    "predicted_species"
                ],
                confidence=prediction_result["confidence"],
                top5_predictions=prediction_result[
                    "top_5_predictions"
                ],
                image_path=relative_image_path
            )

        except Exception:
            # Avoid leaving an image without a database record.
            saved_image_path.unlink(missing_ok=True)
            raise

        print(
            f"Prediction saved. ID: {prediction_id}",
            flush=True
        )

        # --------------------------------------
        # H. RETURN COMPLETE RESPONSE
        # --------------------------------------

        return {
            "success": True,
            "prediction_id": prediction_id,
            "filename": file.filename,
            "image_url": f"/history-images/{unique_filename}",

            "predicted_species": (
                prediction_result["predicted_species"]
            ),

            "confidence": (
                prediction_result["confidence"]
            ),

            "top_5_predictions": (
                prediction_result["top_5_predictions"]
            ),

            "gradcam": {
                "predicted_species": predicted_species,
                "confidence": round(confidence, 2),
                "original_image": visualizations["original_image"],
                "heatmap_image": visualizations["heatmap_image"],
                "overlay_image": visualizations["overlay_image"]
            }
        }

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    except Exception as error:

        print(
            f"Prediction / Grad-CAM error: {error}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:
        await file.close()

# ==========================================
# 11. GET PREDICTION HISTORY
# ==========================================

@app.get("/api/history")
def read_prediction_history():

    try:

        history = get_prediction_history()

        # Add a browser-accessible image URL.
        for record in history:

            image_path = record.get("image_path")

            if image_path:
                image_filename = Path(image_path).name

                record["image_url"] = (
                    f"/history-images/{image_filename}"
                )
            else:
                record["image_url"] = None

        return {
            "success": True,
            "total_records": len(history),
            "history": history
        }

    except Exception as error:

        print(
            f"History retrieval error: {error}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history."
        )

# ==========================================
# 12. DELETE PREDICTION HISTORY
# ==========================================

@app.delete("/api/history/{prediction_id}")
def remove_prediction(prediction_id: int):

    if prediction_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid prediction ID."
        )

    try:

        deleted = delete_prediction(prediction_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Prediction history record not found."
            )

        return {
            "success": True,
            "message": "Prediction history deleted successfully.",
            "prediction_id": prediction_id
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            f"History deletion error: {error}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete prediction history."
        )



# ==========================================
# BIRD DETAILS API
# ==========================================

@app.get("/api/birds/{species_name}")
def get_bird_details(species_name: str):

    result = bird_details_service.get_bird_details(
        species_name
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Bird details not found for: {species_name}"
        )

    return result

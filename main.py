import os
import sys
import uuid
from datetime import datetime
from typing import List

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import engine, Base, get_db
from models import Prediction
from predictor import predict_tree_health

# Create database tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Urban Tree Health Classification API",
    description="AI-powered API for classifying urban tree health using MobileNetV2 and FastAPI.",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup uploads directory
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static directory for served images
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# Pydantic Response Models
class PredictionResponse(BaseModel):
    id: int
    filename: str
    image_url: str
    predicted_class: str
    confidence: float
    recommendation: str
    created_at: str

    class Config:
        from_attributes = True


FRONTEND_DIST = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "dist"))

if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")
    
    @app.get("/")
    def read_root():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def read_root():
        return {
            "status": "online",
            "message": "Urban Tree Health Classification API is running",
            "endpoints": ["/predict (POST)", "/history (GET)", "/uploads/{filename} (GET)"]
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validate image file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed extensions: JPG, JPEG, PNG, WEBP."
        )

    # Read image content
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty."
        )

    # Run AI inference
    try:
        result = predict_tree_health(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image with AI model: {str(e)}"
        )

    # Save image file to uploads folder
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOADS_DIR, unique_filename)
    with open(saved_path, "wb") as f:
        f.write(contents)

    image_url = f"/uploads/{unique_filename}"

    # Create and commit DB record
    db_prediction = Prediction(
        filename=file.filename,
        image_url=image_url,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        recommendation=result["recommendation"],
        created_at=datetime.utcnow()
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return PredictionResponse(
        id=db_prediction.id,
        filename=db_prediction.filename,
        image_url=db_prediction.image_url,
        predicted_class=db_prediction.predicted_class,
        confidence=db_prediction.confidence,
        recommendation=db_prediction.recommendation,
        created_at=db_prediction.created_at.isoformat()
    )


@app.get("/history", response_model=List[PredictionResponse])
def get_history(db: Session = Depends(get_db)):
    records = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    
    formatted = []
    for rec in records:
        formatted.append(PredictionResponse(
            id=rec.id,
            filename=rec.filename,
            image_url=rec.image_url,
            predicted_class=rec.predicted_class,
            confidence=rec.confidence,
            recommendation=rec.recommendation,
            created_at=rec.created_at.isoformat() if rec.created_at else datetime.utcnow().isoformat()
        ))
    return formatted

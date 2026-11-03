from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    image_url = Column(String)
    predicted_class = Column(String, index=True)
    confidence = Column(Float)
    recommendation = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

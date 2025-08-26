import os
from typing import List

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Integer, String, Float
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, Session

from dotenv import load_dotenv

# -----------------------------
# 1) تحميل متغيرات البيئة
# -----------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/demo_db")

# -----------------------------
# 2) إعداد SQLAlchemy
# -----------------------------
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

# -----------------------------
# 3) الموديلات (جداول قاعدة البيانات)
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True, nullable=False)

class Place(Base):
    __tablename__ = "places"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

# إنشاء الجداول (للتطوير السريع). للإنتاج استخدم Alembic.
Base.metadata.create_all(bind=engine)

# -----------------------------
# 4) السكيمات (Pydantic Schemas)
# -----------------------------
class UserCreate(BaseModel):
    name: str
    email: EmailStr

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    class Config:
        from_attributes = True

class PlaceCreate(BaseModel):
    name: str
    latitude: float
    longitude: float

class PlaceOut(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float
    class Config:
        from_attributes = True

# -----------------------------
# 5) تطبيق FastAPI + CORS
# -----------------------------
app = FastAPI(title="FastAPI + PostgreSQL Demo", version="0.1.0")

# CORS (مفتوح أثناء التطوير)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # غيّرها لدوميناتك في الإنتاج
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تبعية: جلسة DB لكل طلب
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# 6) المسارات (Endpoints)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Users (اختياري) ----
@app.post("/users", response_model=UserOut)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id.desc()).all()

# ---- Places ----
@app.post("/places", response_model=PlaceOut)
def create_place(place: PlaceCreate, db: Session = Depends(get_db)):
    p = Place(name=place.name, latitude=place.latitude, longitude=place.longitude)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

@app.get("/places", response_model=List[PlaceOut])
def list_places(db: Session = Depends(get_db)):
    return db.query(Place).order_by(Place.id.desc()).all()

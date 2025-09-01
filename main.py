import os
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Response  # ← زود Response
import json  # ← جديد
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Integer, String, Float, text
from sqlalchemy.orm import (
    sessionmaker,
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
)
from dotenv import load_dotenv

# =============================
# 1) تحميل متغيرات البيئة
# =============================
load_dotenv()

# نقرأ DATABASE_URL من البيئة (Railway)،
# ونوفّر قيمة افتراضية للتجربة محليًا.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/demo_db")

# بعض المنصات القديمة ترجع postgres:// فنحوّلها لـ postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# =============================
# 2) إعداد SQLAlchemy
# =============================
# pool_pre_ping=True لتفادي سقوط الاتصال الصامت
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


# =============================
# 3) الموديلات (جداول قاعدة البيانات)
# =============================
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


# مفيش create_all هنا عشان ما نجبرش الاتصال وقت الاستيراد.
# Base.metadata.create_all(bind=engine)

# =============================
# 4) السكيمات (Pydantic Schemas)
# =============================
class UserCreate(BaseModel):
    name: str
    email: EmailStr


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr

    class Config:
        from_attributes = True  # pydantic v2


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
        from_attributes = True  # pydantic v2


# =============================
# 5) تطبيق FastAPI + CORS
# =============================
app = FastAPI(title="FastAPI + PostgreSQL Demo", version="0.1.0")

# CORS (خلّيه واسع أثناء التطوير، وضيّقه في الإنتاج)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج: ["https://your-domain.com", ...]
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


# =============================
# 6) تهيئة قاعدة البيانات عند الإقلاع
# =============================
@app.on_event("startup")
def on_startup():
    """
    اتصال مبدئي بقاعدة البيانات + إنشاء الجداول بعد تأكد الاتصال.
    لو الاتصال فشل، بنطبع الخطأ فقط بدون إسقاط السيرفر.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # اختبار سريع للاتصال
        Base.metadata.create_all(bind=engine)  # إنشاء الجداول (مؤقتًا بدل Alembic)
        print("✅ DB connected & tables ensured")
    except Exception as e:
        # ما نوقعش السيرفس عشان /health يفضل يرد
        print(f"❌ DB init error: {e}")


# =============================
# 7) المسارات (Endpoints)
# =============================

# مسار خفيف جدًا للتأكد إن السيرفر عايش
@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is alive"}

# مسار صحة لا يلمس DB
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


# ---- Users ----
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
@app.get("/places/geojson", include_in_schema=False)
def places_geojson(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT id, name, latitude, longitude
        FROM public.places
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY id ASC
    """)).mappings().all()

    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "id": int(r["id"]),
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["longitude"]), float(r["latitude"])]
            },
            "properties": {
                "name": r["name"]
            }
        })

    collection = {"type": "FeatureCollection", "features": features}
    return Response(
        content=json.dumps(collection, ensure_ascii=False),
        media_type="application/geo+json"
    )

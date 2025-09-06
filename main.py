# main.py
import os
import uuid
import json
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Integer, String, Float, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, Session
from dotenv import load_dotenv

# =============================
# 1) Env
# =============================
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/demo_db")
if DATABASE_URL.startswith("postgres://"):  # railway/old URIs
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# =============================
# 2) SQLAlchemy
# =============================
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

# =============================
# 3) Models
# =============================
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True, nullable=False)

class Place(Base):
    __tablename__ = "places"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    # حقول إضافية (اختيارية)
    FACILITYID:       Mapped[Optional[str]]   = mapped_column(String(50), nullable=True)
    ELEVATION:        Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    INVERTLEVEL:      Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    GROUNDLEVEL:      Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CONTRACTOR:       Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)
    SUBCONTRACTOR:    Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)
    PROJECTNO:        Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)
    PHASENO:          Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)
    ITEMNO:           Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)
    INSTALLATIONDATE: Mapped[Optional[str]]   = mapped_column(String(50),  nullable=True)
    COVERMATERIAL:    Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)
    X:                Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Y:                Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ARNAME:           Mapped[Optional[str]]   = mapped_column(String(200), nullable=True)
    ENNAME:           Mapped[Optional[str]]   = mapped_column(String(200), nullable=True)
    WALLTHICKNESS:    Mapped[Optional[str]]   = mapped_column(String(200), nullable=True)
    MNOHLESHAPE:      Mapped[Optional[str]]   = mapped_column(String(200), nullable=True)
    DIMENSION:        Mapped[Optional[str]]   = mapped_column(String(200), nullable=True)
    URLLINK:          Mapped[Optional[str]]   = mapped_column(String(255), nullable=True)

    image_url:        Mapped[Optional[str]]   = mapped_column(String(500), nullable=True)

# =============================
# 4) Schemas
# =============================
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
    name: Optional[str] = None
    latitude: float
    longitude: float
    FACILITYID: Optional[str] = None
    ELEVATION: Optional[float] = None
    INVERTLEVEL: Optional[float] = None
    GROUNDLEVEL: Optional[float] = None
    CONTRACTOR: Optional[str] = None
    SUBCONTRACTOR: Optional[str] = None
    PROJECTNO: Optional[str] = None
    PHASENO: Optional[str] = None
    ITEMNO: Optional[str] = None
    INSTALLATIONDATE: Optional[str] = None
    COVERMATERIAL: Optional[str] = None
    X: Optional[float] = None
    Y: Optional[float] = None
    ARNAME: Optional[str] = None
    ENNAME: Optional[str] = None
    WALLTHICKNESS: Optional[str] = None
    MNOHLESHAPE: Optional[str] = None
    DIMENSION: Optional[str] = None
    URLLINK: Optional[str] = None

class PlaceOut(PlaceCreate):
    id: int
    image_url: Optional[str] = None
    class Config:
        from_attributes = True

# =============================
# 5) App + CORS + static
# =============================
app = FastAPI(title="FastAPI + PostgreSQL Demo", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# =============================
# 6) Startup (health + tables + indexes)
# =============================
@app.on_event("startup")
def on_startup():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(bind=engine)

        # فهارس اختيارية لتحسين الأداء عند البحث المكاني (بدون PostGIS)
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_places_lat ON public.places (latitude)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_places_lng ON public.places (longitude)"
            ))
        print("✅ DB connected & tables/indexes ensured")
    except Exception as e:
        print(f"❌ DB init error: {e}")

# =============================
# 7) Endpoints
# =============================
@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is alive"}

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

# ---- Users ----
@app.post("/users", response_model=UserOut)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=409, detail="Email already exists")
    u = User(name=user.name, email=user.email)
    db.add(u); db.commit(); db.refresh(u)
    return u

@app.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id.desc()).all()

# ---- Places (JSON) ----
@app.post("/places", response_model=PlaceOut)
def create_place(place: PlaceCreate, db: Session = Depends(get_db)):
    p = Place(**place.model_dump())
    db.add(p); db.commit(); db.refresh(p)
    return p

@app.get("/places", response_model=List[PlaceOut])
def list_places(db: Session = Depends(get_db)):
    return db.query(Place).order_by(Place.id.desc()).all()

# ---- Places + Image (multipart) ----
@app.post("/places/upload", response_model=PlaceOut)
async def create_place_with_image(
    name: Optional[str] = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),

    FACILITYID: Optional[str] = Form(None),
    ELEVATION: Optional[float] = Form(None),
    INVERTLEVEL: Optional[float] = Form(None),
    GROUNDLEVEL: Optional[float] = Form(None),
    CONTRACTOR: Optional[str] = Form(None),
    SUBCONTRACTOR: Optional[str] = Form(None),
    PROJECTNO: Optional[str] = Form(None),
    PHASENO: Optional[str] = Form(None),
    ITEMNO: Optional[str] = Form(None),
    INSTALLATIONDATE: Optional[str] = Form(None),
    COVERMATERIAL: Optional[str] = Form(None),
    X: Optional[float] = Form(None),
    Y: Optional[float] = Form(None),
    ARNAME: Optional[str] = Form(None),
    ENNAME: Optional[str] = Form(None),
    WALLTHICKNESS: Optional[str] = Form(None),
    MNOHLESHAPE: Optional[str] = Form(None),
    DIMENSION: Optional[str] = Form(None),
    URLLINK: Optional[str] = Form(None),

    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(await image.read())
    image_url = f"/uploads/{fname}"

    p = Place(
        name=name, latitude=latitude, longitude=longitude,
        FACILITYID=FACILITYID, ELEVATION=ELEVATION, INVERTLEVEL=INVERTLEVEL,
        GROUNDLEVEL=GROUNDLEVEL, CONTRACTOR=CONTRACTOR, SUBCONTRACTOR=SUBCONTRACTOR,
        PROJECTNO=PROJECTNO, PHASENO=PHASENO, ITEMNO=ITEMNO, INSTALLATIONDATE=INSTALLATIONDATE,
        COVERMATERIAL=COVERMATERIAL, X=X, Y=Y, ARNAME=ARNAME, ENNAME=ENNAME,
        WALLTHICKNESS=WALLTHICKNESS, MNOHLESHAPE=MNOHLESHAPE, DIMENSION=DIMENSION, URLLINK=URLLINK,
        image_url=image_url
    )
    db.add(p); db.commit(); db.refresh(p)
    return p

# ---- GeoJSON للتطبيق/الخريطة ----
@app.get("/places/geojson", include_in_schema=False)
def places_geojson(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT id, name, latitude, longitude, image_url
        FROM public.places
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY id ASC
    """)).mappings().all()

    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "id": int(r["id"]),
            "geometry": {"type": "Point", "coordinates": [float(r["longitude"]), float(r["latitude"])]},
            "properties": {"name": r["name"], "image_url": r["image_url"]},
        })
    collection = {"type": "FeatureCollection", "features": features}
    return Response(content=json.dumps(collection, ensure_ascii=False), media_type="application/geo+json")

# ---- أقرب أماكن (بدون PostGIS) ----
@app.get("/places/near")
def get_near_places(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    radius_m: int = Query(2000, ge=1, le=100000, description="Search radius in meters"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    يرجّع أقرب الأماكن مترتّبة بالمسافة (متر) باستخدام معادلة Haversine.
    بنستخدم Bounding Box أولًا ثم نحسب المسافة بدقة لتسريع الاستعلام.
    """
    # درجة تقريبية للمتر (عند خط الاستواء ≈ 111.32 كم لكل درجة)
    deg = radius_m / 111_320.0
    sql = text("""
        SELECT id, name, latitude, longitude, image_url,
            (6371000 * acos(LEAST(1,
                cos(radians(:lat)) * cos(radians(latitude)) *
                cos(radians(longitude) - radians(:lng)) +
                sin(radians(:lat)) * sin(radians(latitude))
            ))) AS dist_m
        FROM public.places
        WHERE latitude  BETWEEN :lat - :deg AND :lat + :deg
          AND longitude BETWEEN :lng - :deg AND :lng + :deg
        ORDER BY dist_m
        LIMIT :limit
    """)
    rows = db.execute(sql, {"lat": lat, "lng": lng, "deg": deg, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]

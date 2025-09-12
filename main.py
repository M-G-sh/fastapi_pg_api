# main.py
import os
import uuid
import json
import logging
import traceback
from typing import List, Optional, Tuple

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Response, Query, Request, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Integer, String, Float, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, Session
from dotenv import load_dotenv

# =============================
# 1) Env
# =============================
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/demo_db")
if DATABASE_URL.startswith("postgres://"):  # railway old URIs
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # اختياري لتنظيف الداتا

# مسار رفع الملفات:
UPLOAD_DIR = os.getenv("UPLOAD_DIR")
if not UPLOAD_DIR:
    # إذا شغال على Railway استخدم /tmp
    if os.getenv("RAILWAY_ENVIRONMENT"):
        UPLOAD_DIR = "/tmp/uploads"
    else:
        UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

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
# 5) App + CORS + static + logging
# =============================
app = FastAPI(title="FastAPI + PostgreSQL Demo", version="0.6.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

@app.middleware("http")
async def error_logger(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error("Unhandled error on %s %s: %s\n%s",
                     request.method, request.url.path, e, traceback.format_exc())
        return Response(
            content=json.dumps({"detail": "internal_error", "error": str(e)}),
            media_type="application/json",
            status_code=500,
        )

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================
# 6) Startup (health + tables + indexes + checks)
# =============================
@app.on_event("startup")
def on_startup():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(bind=engine)

        # فهارس + قيود CHECK
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_places_lat ON public.places (latitude)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_places_lng ON public.places (longitude)"))
            try:
                conn.execute(text("ALTER TABLE public.places ADD CONSTRAINT chk_lat CHECK (latitude BETWEEN -90 AND 90)"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE public.places ADD CONSTRAINT chk_lng CHECK (longitude BETWEEN -180 AND 180)"))
            except Exception:
                pass

            # فهرس مكاني ديناميكي لجدول piepe (لو لقى عمود هندسي)
            conn.execute(text("""
                DO $$
                DECLARE gcol text;
                BEGIN
                  SELECT f_geometry_column
                  INTO gcol
                  FROM public.geometry_columns
                  WHERE f_table_schema='public' AND f_table_name='piepe'
                  LIMIT 1;

                  IF gcol IS NOT NULL THEN
                    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_piepe_geom ON public.piepe USING GIST (%I)', gcol);
                  END IF;
                END$$;
            """))

        print(f"✅ DB connected & tables/indexes ensured | UPLOAD_DIR={UPLOAD_DIR}")
    except Exception as e:
        print(f"❌ DB init error: {e}")

# =============================
# 7) Helpers
# =============================
def validate_latlng(lat: float, lng: float):
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        raise HTTPException(status_code=422, detail="Invalid latitude/longitude")

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

# =============================
# 8) Endpoints (Root/Health/Users/Places)
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
    db.add(u)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(u)
    return u

@app.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id.desc()).all()

# ---- Places (JSON) ----
@app.post("/places", response_model=PlaceOut)
def create_place(place: PlaceCreate, db: Session = Depends(get_db)):
    p = Place(**place.model_dump())
    db.add(p)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(p)
    return p

@app.get("/places", response_model=List[PlaceOut])
def list_places(db: Session = Depends(get_db)):
    return (
        db.query(Place)
        .filter(
            Place.latitude  >= -90,
            Place.latitude  <= 90,
            Place.longitude >= -180,
            Place.longitude <= 180,
        )
        .order_by(Place.id.desc())
        .all()
    )

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
    validate_latlng(latitude, longitude)

    filename = image.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    total = 0
    try:
        with open(fpath, "wb") as f:
            while True:
                chunk = await image.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
                f.write(chunk)
    except HTTPException:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass
        raise
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
    db.add(p)
    try:
        db.commit()
    except Exception:
        db.rollback()
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass
        raise
    db.refresh(p)
    return p

# ---- Places GeoJSON (Points)
@app.get("/places/geojson", include_in_schema=False)
def places_geojson(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT id, name, latitude, longitude, image_url
        FROM public.places
        WHERE latitude BETWEEN -90 AND 90
          AND longitude BETWEEN -180 AND 180
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
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

# ---- Places Near (no PostGIS)
@app.get("/places/near")
def get_near_places(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    radius_m: int = Query(2000, ge=1, le=100000, description="Search radius in meters"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    validate_latlng(lat, lng)
    deg = radius_m / 111_320.0
    sql = text("""
        SELECT id, name, latitude, longitude, image_url,
            (6371000 * acos(LEAST(1,
                cos(radians(:lat)) * cos(radians(latitude)) *
                cos(radians(longitude) - radians(:lng)) +
                sin(radians(:lat)) * sin(radians(latitude))
            ))) AS dist_m
        FROM public.places
        WHERE latitude BETWEEN -90 AND 90
          AND longitude BETWEEN -180 AND 180
          AND latitude  BETWEEN :lat - :deg AND :lat + :deg
          AND longitude BETWEEN :lng - :deg AND :lng + :deg
        ORDER BY dist_m
        LIMIT :limit
    """)
    rows = db.execute(sql, {"lat": lat, "lng": lng, "deg": deg, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]

# ---- Admin cleanup
@app.post("/admin/places/cleanup")
def cleanup_invalid_coords(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
    db: Session = Depends(get_db),
):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    res = db.execute(text("""
        UPDATE public.places
        SET latitude = NULL, longitude = NULL
        WHERE NOT (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)
        RETURNING id
    """))
    ids = [r[0] for r in res.fetchall()]
    db.commit()
    return {"cleaned": ids, "count": len(ids)}

# =============================
# 9) Piepe (PostGIS GeoJSON + PATCH)
# =============================

PIEPE_EDITABLE_FIELDS = [
    "FACILITYID","ELEVATION","INVERTLEVEL","GROUNDLEVEL","CONTRACTOR","SUBCONTRACTOR",
    "PROJECTNO","PHASENO","ITEMNO","INSTALLATIONDATE","COVERMATERIAL","X","Y",
    "ARNAME","ENNAME","WALLTHICKNESS","MNOHLESHAPE","DIMENSION","URLLINK"
]

PIEPE_ID_CANDIDATES   = ["id", "fid", "ogc_fid", "gid", "objectid"]
PIEPE_GEOM_CANDIDATES = ["geom", "wkb_geometry", "the_geom"]

def _valid_bbox(bbox: Optional[str]) -> Optional[List[float]]:
    if not bbox:
        return None
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            return None
        minx, miny, maxx, maxy = parts
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
            return None
        if minx >= maxx or miny >= maxy:
            return None
        return [minx, miny, maxx, maxy]
    except Exception:
        return None

def _piepe_resolve_cols(db: Session) -> Tuple[str, str, str, str]:
    # 1) id column
    id_col = db.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='piepe'
          AND column_name = ANY(:cands)
        LIMIT 1
    """), {"cands": PIEPE_ID_CANDIDATES}).scalar()

    # 2) geom column + udt_name
    geom_row = db.execute(text("""
        SELECT c.column_name, c.udt_name
        FROM information_schema.columns c
        WHERE c.table_schema='public' AND c.table_name='piepe'
          AND c.column_name = ANY(:cands)
        LIMIT 1
    """), {"cands": PIEPE_GEOM_CANDIDATES}).mappings().first()

    if not id_col:
        raise HTTPException(status_code=500, detail="piepe id column not found (id/fid/ogc_fid/gid/objectid)")
    if not geom_row:
        raise HTTPException(status_code=500, detail="piepe geometry column not found (geom/wkb_geometry/the_geom)")

    geom_col = geom_row["column_name"]
    udt_name = (geom_row["udt_name"] or "").lower()

    # 3) استنتاج SRID من أول صف (باستعلام منفصل لكل نوع)
    if udt_name == "geometry":
        srid_row = db.execute(text(f"""
            SELECT NULLIF(ST_SRID("{geom_col}"),0) AS srid
            FROM public.piepe
            WHERE "{geom_col}" IS NOT NULL
            LIMIT 1
        """)).mappings().first()
    else:
        srid_row = db.execute(text(f"""
            SELECT NULLIF(ST_SRID(ST_GeomFromWKB("{geom_col}"::bytea)),0) AS srid
            FROM public.piepe
            WHERE "{geom_col}" IS NOT NULL
            LIMIT 1
        """)).mappings().first()

    srid = int((srid_row or {}).get("srid") or 4326)

    # 4) تعبيرات الإخراج والفلترة (نحوّل دايمًا لـ 4326)
    if udt_name == "geometry":
        geom_in_4326 = f"""
            CASE WHEN ST_SRID("{geom_col}")=4326
                 THEN "{geom_col}"
                 ELSE ST_Transform("{geom_col}", 4326)
            END
        """
    else:
        geom_in_4326 = f"""
            ST_Transform(
                ST_SetSRID(
                    ST_GeomFromWKB("{geom_col}"::bytea),
                    {srid}
                ),
                4326
            )
        """

    geom_json_expr   = f"ST_AsGeoJSON({geom_in_4326})::json"
    bbox_filter_expr = f"ST_Intersects({geom_in_4326}, ST_MakeEnvelope(:minx,:miny,:maxx,:maxy,4326))"

    return id_col, geom_col, geom_json_expr, bbox_filter_expr

@app.get("/piepe/geojson")
def piepe_geojson(
    bbox: Optional[str] = Query(None, description="minLng,minLat,maxLng,maxLat (EPSG:4326)"),
    limit: int = Query(5000, ge=1, le=100000),
    db: Session = Depends(get_db),
):
    id_col, geom_col, geom_json_expr, bbox_filter_expr = _piepe_resolve_cols(db)

    where = "TRUE"
    params = {"limit": limit}
    bbox_vals = _valid_bbox(bbox)
    if bbox_vals:
        where = bbox_filter_expr
        params.update({"minx": bbox_vals[0], "miny": bbox_vals[1], "maxx": bbox_vals[2], "maxy": bbox_vals[3]})

    sql = text(f"""
        WITH q AS (
          SELECT
            "{id_col}"  AS fid,
            "{geom_col}" AS geom,
            (to_jsonb(p) - '{geom_col}' - '{id_col}') AS props
          FROM public.piepe AS p
          WHERE {where}
          ORDER BY "{id_col}"
          LIMIT :limit
        )
        SELECT
          fid,
          {geom_json_expr} AS geometry,
          props AS properties
        FROM q;
    """)
    rows = db.execute(sql, params).mappings().all()

    features = [{
        "type": "Feature",
        "id": int(r["fid"]),
        "geometry": r["geometry"],
        "properties": dict(r["properties"]),
    } for r in rows]

    return Response(
        content=json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        media_type="application/geo+json",
    )

@app.patch("/piepe/{fid}")
def piepe_patch(
    fid: int,
    patch: dict,
    db: Session = Depends(get_db),
):
    if not patch:
        raise HTTPException(status_code=400, detail="Empty patch")

    updates = {k: v for k, v in patch.items() if k in PIEPE_EDITABLE_FIELDS}
    if not updates:
        raise HTTPException(status_code=400, detail="No editable fields in payload")

    id_col, _, _, _ = _piepe_resolve_cols(db)

    set_clauses = []
    params = {"fid": fid}
    i = 0
    for k, v in updates.items():
        i += 1
        p = f"v{i}"
        set_clauses.append(f'"{k}" = :{p}')
        params[p] = v

    sql = text(f'UPDATE public.piepe SET {", ".join(set_clauses)} WHERE "{id_col}" = :fid RETURNING "{id_col}";')
    res = db.execute(sql, params).first()
    if not res:
        db.rollback()
        raise HTTPException(status_code=404, detail="Feature not found")
    db.commit()
    return {"ok": True, "id": fid, "updated": list(updates.keys())}

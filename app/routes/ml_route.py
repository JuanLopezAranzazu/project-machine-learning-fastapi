from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import joblib
from ..utils import image_utils

# Ruta para el modelo de machine learning

NAME_ROUTE = "machine-learning"

router = APIRouter(
  prefix=f"/{NAME_ROUTE}",
  tags=[NAME_ROUTE]
)

# Cargar el modelo
with open("model.pkl", "rb") as f:
  model = joblib.load(f)

@router.get("/")
async def read_root():
  return {"Hello": "ML Model"}

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
  # Verificar que el archivo sea una imagen
  if file.content_type not in ["image/jpeg", "image/png"]:
    raise HTTPException(status_code=400, detail="Invalid image format")

  # Leer y procesar la imagen
  image = await file.read()
  image = Image.open(io.BytesIO(image))
  processed_image = image_utils.preprocess_image(image)

  # Realizar la predicción
  prediction = model.predict(processed_image)

  return JSONResponse(content={"number": int(prediction[0])})


from PIL import Image, ImageOps
import numpy as np

# Funcion para preprocesar la imagen
def preprocess_image(image: Image.Image) -> np.ndarray:
  image = image.convert("L")
  image = ImageOps.invert(image)
  image = image.resize((28, 28))
  image = np.array(image).reshape(1, -1)
  return image

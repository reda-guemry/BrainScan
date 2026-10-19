

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

VALID_EXTENSIONS = ('.jpeg' , '.jpg' , '.png' , '.bmp' )
IMG_SIZE = (224, 224)
ROOT = ROOT_DIR / 'data' 
SEED = 42 
BATCH_SIZE = 32 
EPOCHS = 20
LEARNING_RATE = 0.001

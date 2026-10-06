import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from corruption import apply_corruption


def test_corruption(image_path: str, corruption_type: str, severity: int = 3):
    img = Image.open(image_path).convert("RGB")
    corrupted_img = apply_corruption(img, corruption_type=corruption_type, severity=severity)
    corrupted_img.show()


if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\Regen\Machine Learning\German Dataset\Meta\25.png"
    CORRUPTION_TYPE = "fog"
    SEVERITY = 3  # 1-5

    test_corruption(IMAGE_PATH, CORRUPTION_TYPE, severity=SEVERITY)

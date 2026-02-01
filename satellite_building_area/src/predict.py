import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .models import load_models
from skimage.morphology import remove_small_objects
from .config import DEVICE, SEG, SCALE

def predict_area(image, auto_scale, manual_scale):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 1. Квадрат и float32
    h, w, _ = image.shape
    
    side = min(h, w)
    image = cv2.resize(image, (side, side))
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)

    # 2. Сегментация
    with torch.no_grad():
        logits, bottleneck, skips = SEG(image_tensor)
        prob_mask = F.sigmoid(logits).squeeze().cpu().numpy()  # [256, 256] 0-1
    # 3. Масштаб
    if auto_scale:
        with torch.no_grad():
            scale = SCALE(bottleneck, skips).item()
    else:
        scale = manual_scale

    # 4. Бинаризация
    binary_mask = (prob_mask > 0.5).astype(bool)

    # 5. Очистка
    binary_mask = clean_mask(binary_mask, min_area=30)

    # 6. Площадь
    n_pixels = binary_mask.sum()
    area_m2 = n_pixels * (scale ** 2)

    # 7. Визуализация
    overlay = image.copy()
    alpha = 0.5
    green = [0, 255, 0]
    for c in range(3):
        overlay[:, :, c] = np.where(binary_mask,
                                    (1 - alpha) * overlay[:, :, c] + alpha * green[c],
                                    overlay[:, :, c])

    return overlay, f"{area_m2:.2f} м²", f"{scale:.3f} м/пикс"

def clean_mask(mask, min_area=10):
    """
    Удаляет мелкие объекты < min_area (в пикселях)
    mask: [H, W] bool
    """
    return remove_small_objects(mask, max_size=min_area - 1)
import cv2
import numpy as np
from PIL import Image

STANDARD_SIZE = (512, 512)
BLUR_KERNEL = (5, 5)  # Mírně větší jádro pro lepší redukci šumu
CANNY_THRESHOLDS = (30, 150)
DILATE_ITER = 2

def preprocess_image(image_pil):
    """
    Převede PIL obrázek do formátu OpenCV a změní velikost.
    """
    # Převod PIL obrázku na numpy pole (formát OpenCV)
    image_np = np.array(image_pil)
    
    # Převod RGB na BGR (OpenCV používá BGR)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    # Změna velikosti pro konzistentní zpracování (Zachování poměru stran)
    orig_h, orig_w = image_np.shape[:2]
    
    # Výpočet měřítka tak, aby se vešlo do STANDARD_SIZE[0] (např. 512) na nejdelší straně
    scale = STANDARD_SIZE[0] / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = cv2.resize(image_np, (new_w, new_h))
    
    return resized, (orig_w, orig_h)

def detect_regions(image_cv, param1=50, param2=30, min_radius=10, max_radius=100):
    """
    Detekuje oblasti podobné mincím pomocí Houghovy transformace kružnic.
    Vrací: seznam n-tic (x, y, r) a debug obrázek.
    """
    # Předzpracování
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Snížené rozmazání pro lepší zachování hran (bylo 9,9, 2)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Houghova transformace kružnic
    # method=cv2.HOUGH_GRADIENT
    # dp=1: Inverzní poměr rozlišení akumulátoru k rozlišení obrázku.
    # minDist=param: Minimální vzdálenost mezi středy detekovaných kružnic.
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=min_radius * 2,  # Předpoklad, že se mince nepřekrývají více než o tuto hodnotu
        param1=param1, # Vyšší práh pro Canny detektor hran
        param2=param2, # Práh akumulátoru (nižší = více kružnic, vyšší = méně/lepší)
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    candidates = []
    debug_image = image_cv.copy()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Seřazení kružnic podle poloměru (SESTUPNĚ) pro preferenci větších vnějších kružnic.
        # Toto je KRITICKÉ pro bimetalické mince (50 Kč), kde chceme celou minci,
        # ne jen vnitřní prstenec.
        sorted_indices = np.argsort(circles[0, :, 2])[::-1]
        sorted_circles = circles[0, sorted_indices, :]

        selected_circles = []
        for i in range(sorted_circles.shape[0]):
            x, y, r = int(sorted_circles[i, 0]), int(sorted_circles[i, 1]), int(sorted_circles[i, 2])
            
            is_duplicate = False
            for (sx, sy, sr) in selected_circles:
                # Výpočet vzdálenosti mezi středy
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                
                # Kontrola významného překryvu
                # Pokud jsou středy blíže než větší poloměr, jedna je pravděpodobně uvnitř druhé
                # Použití faktoru (např. 0.85) pro povolení mírného překryvu, ale zamítnutí soustředných
                max_r = max(r, sr)
                if dist < max_r * 0.85: 
                     is_duplicate = True
                     break
            
            if not is_duplicate:
                selected_circles.append((x, y, r))
                candidates.append((x, y, r))
                # Vykreslení vnější kružnice
                cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
                # Vykreslení středu
                cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
            
    return candidates, debug_image

def extract_coin_image(image_cv, center_x, center_y, radius, target_size=(128, 128)):
    """
    Extrahuje čtvercový výřez kolem mince a změní velikost pro CNN.
    """
    h, w = image_cv.shape[:2]
    
    # Přidání odsazení pro zachycení hrany
    padding = int(radius * 0.3)
    x1 = max(0, center_x - radius - padding)
    y1 = max(0, center_y - radius - padding)
    x2 = min(w, center_x + radius + padding)
    y2 = min(h, center_y + radius + padding)
    
    crop = image_cv[y1:y2, x1:x2]
    
    if crop.size == 0:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
    crop_resized = cv2.resize(crop, target_size)
    
    # Převod zpět na RGB pro model (předpoklad modelu trénovaného na RGB)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    
    return crop_rgb

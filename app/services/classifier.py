from __future__ import annotations

from collections import deque
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, ema_alpha: float = 0.2, max_history: int = 2000, recalc_interval: int = 30):
        self.ema_alpha = ema_alpha
        self.recalc_interval = recalc_interval
        
        # CORRECCIÓN: Usamos un deque con un tamaño máximo para evitar consumo infinito de memoria
        # y permitir que el clasificador se adapte a sutiles cambios de luz en partidos largos.
        self.all_colors: deque[list[float]] = deque(maxlen=max_history)
        
        self.team_1_color: list[float] | None = None
        self.team_2_color: list[float] | None = None
        self.frame_count = 0

    def update(self, player_colors: list[list[float]]) -> None:
        self.all_colors.extend(player_colors)
        self.frame_count += 1

        # Requerimos una base mínima de datos histórica para empezar a clasificar
        if len(self.all_colors) < 40:
            return

        should_recalc = self.team_1_color is None or self.frame_count % self.recalc_interval == 0
        if not should_recalc:
            return

        # Ajustamos a los datos actuales guardados en el deque
        colors_array = np.array(list(self.all_colors))
        kmeans = KMeans(n_clusters=2, random_state=42, init="k-means++", n_init=5).fit(colors_array)
        c1, c2 = kmeans.cluster_centers_
        c1l = c1.tolist()
        c2l = c2.tolist()

        # CORRECCIÓN DE ESTABILIDAD: Mantener la identidad histórica basada exclusivamente en la
        # distancia euclidiana del color previo para evitar que los equipos se intercambien (flickering)
        if self.team_1_color is not None and self.team_2_color is not None:
            d1_to_t1 = np.linalg.norm(np.array(c1l) - np.array(self.team_1_color))
            d2_to_t1 = np.linalg.norm(np.array(c2l) - np.array(self.team_1_color))
            
            # Si el centro 2 está más cerca del equipo 1 histórico, los invertimos para mantener consistencia
            if d2_to_t1 < d1_to_t1:
                c1l, c2l = c2l, c1l
        else:
            # Inicialización por primera vez: ordenamos por luminancia por defecto inicial
            def luminance(lab_color):
                # Si viene en formato LAB (L es el primer canal), o si es RGB clásico:
                return lab_color[0]
            if luminance(c2l) > luminance(c1l):
                c1l, c2l = c2l, c1l

        # Aplicación del suavizado temporal (EMA)
        if self.team_1_color is None:
            self.team_1_color = c1l
            self.team_2_color = c2l
        else:
            a = self.ema_alpha
            self.team_1_color = [a * c1l[i] + (1 - a) * self.team_1_color[i] for i in range(3)]
            self.team_2_color = [a * c2l[i] + (1 - a) * self.team_2_color[i] for i in range(3)]

    def assign(self, player_color: list[float]) -> int:
        if self.team_1_color is None or self.team_2_color is None:
            return 0
        d1 = np.linalg.norm(np.array(player_color) - np.array(self.team_1_color))
        d2 = np.linalg.norm(np.array(player_color) - np.array(self.team_2_color))
        return 0 if d1 < d2 else 1

    @property
    def colors_ready(self) -> bool:
        return self.team_1_color is not None and self.team_2_color is not None

    def reset(self) -> None:
        self.all_colors.clear()
        self.team_1_color = None
        self.team_2_color = None
        self.frame_count = 0
def get_player_team_color(player_crop: np.ndarray) -> np.ndarray | None:
    """
    Extrae el color dominante del torso del jugador, eliminando el fondo (césped)
    y las zonas de alto contraste (dorsales, logos, publicidad).
    """
    if player_crop.shape[0] < 10 or player_crop.shape[1] < 10:
        return None

    # 1. Tomamos el torso del jugador (del 15% al 45% de la altura de la caja)
    h, w = player_crop.shape[:2]
    y_start = int(h * 0.15)
    y_end = int(h * 0.45)
    margin_x = int(w * 0.15)
    
    crop_shirt = player_crop[y_start:y_end, margin_x:w-margin_x]
    if crop_shirt.size == 0:
        return None

    # 2. FILTRO 1: Eliminar el césped usando HSV
    hsv_shirt = cv2.cvtColor(crop_shirt, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_shirt, lower_green, upper_green)

    # 3. FILTRO 2: Detectar y eliminar dorsales / publicidad (Zonas de alto contraste)
    gray_shirt = cv2.cvtColor(crop_shirt, cv2.COLOR_BGR2GRAY)
    
    # Calculamos el gradiente (bordes) usando Sobel en X e Y
    sobelx = cv2.Sobel(gray_shirt, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_shirt, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    # Normalizamos el gradiente a rango 0-255
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
    
    # Creamos una máscara para los bordes afilados (dorsales, letras, costuras duras)
    # Un umbral entre 40 y 70 suele aislar perfectamente las letras y números
    _, high_contrast_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Expandimos un poco los bordes detectados (dilatación) para asegurarnos de borrar el interior de los números
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    high_contrast_mask = cv2.dilate(high_contrast_mask, kernel, iterations=1)

    # 4. Combinar ambas máscaras: Queremos píxeles que NO sean verdes Y que NO sean dorsales
    bad_pixels_mask = cv2.bitwise_or(green_mask, high_contrast_mask)
    good_pixels_mask = cv2.bitwise_not(bad_pixels_mask)

    # Extraer los píxeles limpios de la camiseta
    player_pixels_bgr = crop_shirt[good_pixels_mask > 0]

    # Guardacoches: Si nos quedamos sin píxeles (ej. camiseta con rayas muy finas), 
    # relajamos el filtro usando solo la máscara de césped original
    if len(player_pixels_bgr) < 15:
        player_pixels_bgr = crop_shirt[green_mask == 0]
    if len(player_pixels_bgr) < 15:
        player_pixels_bgr = crop_shirt.reshape(-1, 3)

    # 5. Convertir píxeles limpios a LAB para clustering perceptual
    player_pixels_bgr_img = np.uint8([player_pixels_bgr])
    player_pixels_lab = cv2.cvtColor(player_pixels_bgr_img, cv2.COLOR_BGR2LAB)[0]

    # 6. KMeans para obtener el color real de la tela
    features = player_pixels_lab.astype(np.float32)
    kmeans = KMeans(n_clusters=2, random_state=42, init="k-means++", n_init=2).fit(features)
    
    labels = kmeans.labels_
    c0_count = np.sum(labels == 0)
    c1_count = np.sum(labels == 1)

    dominant_lab = kmeans.cluster_centers_[0] if c0_count > c1_count else kmeans.cluster_centers_[1]

    # 7. Convertir de vuelta a BGR
    shirt_bgr = cv2.cvtColor(np.uint8([[dominant_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
    
    return shirt_bgr.astype(np.float64)
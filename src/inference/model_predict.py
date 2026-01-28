# -------------------------------------------------------
# IMPORTACIONES
# -------------------------------------------------------
import supervision as sv
from src.inference.model_loader import load_model
from src.config.dogs_classes import DOGS_CLASSES

import time

# -------------------------------------------------------
# CARGANDO MODELO FINE TUNING
# -------------------------------------------------------
try:
    model = load_model(
        model_type='small', 
        weights_path='src/models/small/checkpoint_best_total.pth', 
        resolution_image=640
    )

    print("[INFO] Modelo cargado con los parámetros correctamente")
    
except Exception as e:
    raise RuntimeError(f"[ERROR] {e}")
    #print(f"Error al cargar el modelo: {e}")
    #exit(1)

# -------------------------------------------------------
# APLICANDO CONFIGURACIONES DE VISUALIZACIÓN BBOX
# -------------------------------------------------------
color_bounding_boxes = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

resolution_frame = (640,640)
text_scale       = sv.calculate_optimal_text_scale(resolution_wh=resolution_frame)
thickness        = sv.calculate_optimal_line_thickness(resolution_wh=resolution_frame)
bbox_annotator   = sv.BoxAnnotator(color=color_bounding_boxes, thickness=thickness)

label_annotator = sv.LabelAnnotator(
    color=color_bounding_boxes,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    smart_position=True
)

# -------------------------------------------------------
# LÓGICA PARA EJECUTAR PREDICCIÓN O INFERENCIA
# -------------------------------------------------------
def execute_predict( frame_webcam ):
    """
        Ejecuta la predicción del modelo y aplica las configuraciones
        para cada bounding box.

        Args:
            - Frame_webcam: frame que captura la cámara cada cierto tiempo
            - inference_model_time: tiempo que duró la inferencia del modelo
    """

    # -------------- Etapa 1: Preprocesamiento -------------- #
    rgb_frame = frame_webcam[:,:,::-1].copy()

    # -------------- Etapa 2: Inferencia del modelo -------------- #
    inference_start = time.perf_counter()
    predictions     = model.predict( rgb_frame, threshold=0.5 )
    inference_end   = time.perf_counter()

    inference_model_time  = inference_end - inference_start

    # -------------- Etapa 3: Postprocesamiento -------------- #

    # Construyendo cada label para los bounding boxes
    labels = []

    for class_id, confidence in zip( predictions.class_id, predictions.confidence ):
        label = f"{DOGS_CLASSES.get(class_id, "unknown")} {confidence:.2f}"
        labels.append( label )

    # Aplicando configuraciones de bboxes a cada frame (imagen)
    annotated_frame = bbox_annotator.annotate(rgb_frame, predictions)
    annotated_frame = label_annotator.annotate(annotated_frame, predictions, labels)

    # Regresando frame procesado y tiempo de inferencia
    return annotated_frame[:, :, ::-1], inference_model_time

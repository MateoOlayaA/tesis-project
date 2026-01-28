# ---------------------------------------------------
# IMPORTANDO LIBRERÍAS
# ---------------------------------------------------
import cv2

# ---------------------------------------------------
# CONFIGURACIONES INICIALES
# ---------------------------------------------------
def init_configuration():
    """
        Carga las configuraciones iniciales para la cámara web
        y el video de salida.

        Args:
        
        Return:
            - camera: instancia de objeto de la cámara con las configuaciones.
            - out_video: instancia de objeto del video con las configuraciones.
    """
    
    # Activando cámara para leer cada frame
    print("[INFO] Iniciando captura de cámara (esperando dispositivo)")
    camera = cv2.VideoCapture(0)
    print("[INFO] Dispositivo de cámara detectado")

    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    
    DESIRED_WIDTH  = 640
    DESIRED_HEIGHT = 640
    DESIRED_FPS    = 3

    # Solicitando configuraciones según lo que soporte la cámara
    camera.set( cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH )
    camera.set( cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT )
    camera.set( cv2.CAP_PROP_FPS, DESIRED_FPS )

    # Obteniendo las configuraciones reales de la cámara
    frame_width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps     = camera.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Resolución cámara: {frame_width}x{frame_height}")
    print(f"[INFO] FPS solicitados: {DESIRED_FPS} | FPS reales: {real_fps}")

    # Aplicando configuraciones al video de salida
    fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        "src/videos/salida.mp4", 
        fourcc, 
        #real_fps if real_fps > 0 else DESIRED_FPS, # Estableciendo FPS
        DESIRED_FPS,
        (frame_width, frame_height)
    )

    if not out_video.isOpened():
        camera.release()
        raise RuntimeError("No se pudo abrir la cámara")

    print("[INFO] Cámara abierta correctamente para grabación")

    return camera, out_video
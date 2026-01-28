# ---------------------------------------------------
# IMPORTANDO LIBRERÍAS
# ---------------------------------------------------
from src.config.camera_config import init_configuration
from src.inference.model_predict import execute_predict
from src.config import signal_handler

import time

# ---------------------------------------------------
# LÓGICA DE EJECUCIÓN
# ---------------------------------------------------
def execute_process():

    # Iniciando configuraciones para detener proceso
    signal_handler.init_signal_handler()

    # Diccionario que guarda el tiempo en qué se procesó cada frame por el modelo
    times_for_each_frame = {}
    id_frame             = 0

    try:
        # Configuraciones iniciales para cámara y video de salida
        camera, out_video = init_configuration()

        # Loop para capturar cada frame
        print("[INFO] Loop de frames en ejecución")

        while signal_handler.running:

            # Leyendo frame de la cámara en cada instante de tiempo
            success, frame = camera.read()

            # Verificar que la lectura del frame fue correcta
            if not success:
                print("[ERROR] Frame perdido")
                continue
            
            # Aplicando proceso de predicción y el tiempo de duración
            id_frame = id_frame + 1

            inference_start                     = time.perf_counter()
            process_frame, inference_model_time = execute_predict(frame_webcam=frame)
            inference_end                       = time.perf_counter()

            inference_total_time = inference_end - inference_start

            # Guardando el tiempo de duración de la inferencia aplicada sobre el frame
            times_for_each_frame[ f"frame_{ id_frame }" ] = {
                "total": inference_total_time,
                "model": inference_model_time
            }

            # Guardando cada frame para crear el video de salida
            out_video.write( process_frame )

    except RuntimeError as e:
        print(f"[FATAL] {e}")

    finally:
        # Liberar recursos
        camera.release()
        out_video.release()

        # Guardando tiempos de inferencia un archivo .txt
        with open("src/metrics/inference_times.txt", "w", encoding="utf-8") as file:
            file.write(f"Pre: preprocesamiento | Inf: Inferencia | Post: Postprocesamiento \n")

            for key, value in times_for_each_frame.items():
                file.write(
                    f"Frame: {key} | "
                    f"Inferencia total (Pre|Inf|Post) (s): {value['total']:.6f} | "
                    f"Inferencia modelo (s): {value['model']:.6f}\n"
                )

        print("[INFO] Grabación finalizada correctamente")
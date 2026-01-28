# ---------------------------------------------------
# IMPORTANDO LIBRERÍAS
# ---------------------------------------------------
import signal

# ---------------------------------------------------
# CONFIGURACIONES PARA DETENER EL PROCESO
# ---------------------------------------------------
running = True

def stop_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C detectado, cerrando")
    running = False

def init_signal_handler():
    signal.signal(signal.SIGINT, stop_handler)

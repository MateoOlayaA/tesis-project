# -------------------------------------------------------
# IMPORTACIONES
# -------------------------------------------------------
from rfdetr import RFDETRSmall

# -------------------------------------------------------
# LÓGICA PARA CARGAR MODELO FINE TUNING
# -------------------------------------------------------
def load_model( model_type: str, weights_path: str, resolution_image: int = 640 ):
    """
        Carga un modelo RF-DETR fine-tuned y lo optimiza para inferencia.

        Args:
            - model_type: indica la variante del modelo que se cargará.
            - weights_path: ruta donde está el archivo con los pesos fine tuned del modelo.
            - resolution_image: indica el tamaño de las imágenes que recibirá el modelo.

        Returns:
            - model: Modelo RF-DETR listo para la inferencia.
    """
    model_type = model_type.lower()

    if model_type == 'small':
        model_RFDETR = RFDETRSmall(
            pretrain_weights=weights_path,
            resolution=resolution_image
        )

    # Optimizando modelo cargado para la inferencia
    model_RFDETR.optimize_for_inference()

    return model_RFDETR
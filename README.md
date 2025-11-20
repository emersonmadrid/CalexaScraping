# CalexaScraping - Automatización CEJ Poder Judicial

Automatización para búsqueda de expedientes en el CEJ del Poder Judicial Peruano.

## Instalación

1. Clonar repositorio
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar variables opcionales creando `.env` a partir de `.env.example`

## Ejecución

El flujo completo (leer CSV, abrir el navegador, resolver CAPTCHAs y ejecutar las búsquedas) se ejecuta con:

```bash
python -m src.main
```

El archivo `data/inputs/expedientes.csv` contiene los expedientes a procesar. El resultado de cada intento queda registrado en `logs/` y las capturas temporales se guardan en `data/temp/`.

## Arquitectura

- `src/config/settings.py`: carga de configuración y rutas de trabajo.
- `src/models/data_models.py`: dataclasses para expedientes y resultados de CAPTCHA.
- `src/services/captcha_common.py`: captura unificada del CAPTCHA y contratos para los solvers.
- `src/services/captcha_solver_{cnn,ml}.py`: solvers específicos (CNN, patrón + EasyOCR).
- `src/services/captcha_solver.py`: ensemble OCR (Tesseract + EasyOCR) como fallback.
- `src/services/captcha_manager.py`: coordina captura y solvers con reintentos.
- `src/services/form_filler.py`: llena el formulario del CEJ usando Selenium.

Los scripts de entrenamiento (`entrenar_manual.py`, `CaptchaCNNTrainer`, etc.) permiten mejorar los modelos almacenando datos en `data/training/`.

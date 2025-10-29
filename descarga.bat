@echo off
echo ========================================
echo  CONFIGURACION CALEXASCRAPING
echo ========================================
echo.

REM 1. Verificar Python
echo [1/5] Verificando Python...
python --version
if errorlevel 1 (
    echo ERROR: Python no encontrado
    echo Descarga Python desde https://www.python.org/downloads/
    pause
    exit /b 1
)
echo OK - Python encontrado
echo.

REM 2. Crear entorno virtual
echo [2/5] Creando entorno virtual...
if exist venv (
    echo Entorno virtual ya existe
) else (
    python -m venv venv
    echo Entorno virtual creado
)
echo.

REM 3. Activar entorno virtual
echo [3/5] Activando entorno virtual...
call venv\Scripts\activate.bat
echo OK - Entorno activado
echo.

REM 4. Actualizar pip
echo [4/5] Actualizando pip...
python -m pip install --upgrade pip
echo.

REM 5. Instalar dependencias
echo [5/5] Instalando dependencias...
pip install -r requirements.txt
echo.

echo ========================================
echo  INSTALACION COMPLETA
echo ========================================
echo.
echo Proximos pasos:
echo 1. Descargar Tesseract OCR:
echo    https://github.com/UB-Mannheim/tesseract/wiki
echo.
echo 2. Activar entorno virtual:
echo    venv\Scripts\activate
echo.
echo 3. Ejecutar prueba:
echo    python test_optimizado.py
echo.
pause
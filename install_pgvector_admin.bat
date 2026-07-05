@echo off
echo ========================================
echo  Installing pgvector for PostgreSQL 16
echo ========================================
echo.

set SRC=C:\Users\Lenovo\AppData\Local\Temp\pgvector_extract
set DST=C:\Program Files\PostgreSQL\16

echo Copying vector.dll to lib...
copy /Y "%SRC%\lib\vector.dll" "%DST%\lib\vector.dll"
if errorlevel 1 (
    echo ERROR: Failed to copy vector.dll. Make sure you right-clicked and chose "Run as administrator".
    pause
    exit /b 1
)

echo Copying extension files...
xcopy /Y /S "%SRC%\share\extension\*" "%DST%\share\extension\"

echo.
echo ========================================
echo  pgvector installed successfully!
echo ========================================
echo.
echo Now restart PostgreSQL and run:
echo   python setup_db.py
echo.
pause

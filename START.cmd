@echo off

call “C:\OSGeo4W”\bin\o4w_env.bat

call “C:\OSGeo4W”\apps\grass\grass83\etc\env.bat

@echo off

set PYTHONHOME=C:\OSGeo4W\apps\Python312

set PYTHONPATH=%PYTHONPATH%;C:\OSGeo4W\apps\grass\grass83\etc\python;C:\OSGeo4W\apps\qgis-ltr\python;C:\OSGeo4W\apps\qgis-ltr\python\plugins

path %PATH%;C:\OSGeo4W\apps\qgis-ltr\bin

path %PATH%;C:\OSGeo4W\apps\grass\grass83\lib

path %PATH%;C:\OSGeo4W\apps\grass\grass83\bin

path %PATH%;C:\OSGeo4W\apps\Qt5\bin

path %PATH%;C:\OSGeo4W\apps\Python312\Scripts

path %PATH%;C:\OSGeo4W\bin

set GDAL_DATA=C:\OSGeo4W\apps\gdal\share\gdal

set GDAL_DRIVER_PATH=C:\OSGeo4W\apps\gdal\lib\gdalplugins

set GDAL_FILENAME_IS_UTF8=YES

REM set PDAL_DRIVER_PATH=C:\OSGeo4W\apps\pdal\plugins

set GISBASE=C:\OSGeo4W\apps\grass\grass83

set GRASS_PROJSHARE=C:\OSGeo4W\share\proj

set GRASS_PYTHON=C:\OSGeo4W\bin\python3.exe

set OSGEO4W_ROOT=C:\OSGeo4W

set PROJ_LIB=C:\OSGeo4W\share\proj

set PYTHONUTF8=1

set QGIS_PREFIX_PATH=C:\OSGeo4W\apps\qgis-ltr

set QT_PLUGIN_PATH=C:\OSGeo4W\apps\qgis-ltr\qtplugins;C:\OSGeo4W\apps\Qt5\plugins

set VSI_CACHE=TRUE

set VSI_CACHE_SIZE=1000000

set O4W_QT_PREFIX=C:\OSGeo4W\apps\Qt5

set O4W_QT_BINARIES=C:\OSGeo4W\apps\Qt5\bin

set O4W_QT_PLUGINS=C:\OSGeo4W\apps\Qt5\plugins

set O4W_QT_LIBRARIES=C:\OSGeo4W\apps\Qt5\lib

set O4W_QT_TRANSLATIONS=C:\OSGeo4W\apps\Qt5\translations

set O4W_QT_HEADERS=C:\OSGeo4W\apps\Qt5\include

set QT_QPA_PLATFORM_PLUGIN_PATH=C:\OSGeo4W\apps\Qt5\plugins\platforms

set QT_PLUGIN_PATH=C:\OSGeo4W\apps\qgis\qtplugins;C:\OSGeo4W\apps\Qt5\plugins

set QGIS_WIN_APP_NAME=QGIS 3.34\QGIS Desktop 3.34.6

set SSL_CERT_DIR=C:\OSGeo4W\apps\openssl\certs

set SSL_CERT_FILE=C:\OSGeo4W\bin\curl-ca-bundle.crt

start "" "F:\Progers\PyCharm Community Edition 2023.1.1\bin\pycharm64.exe"


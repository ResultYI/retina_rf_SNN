@echo off
setlocal
cd /d "%~dp0"
set "DOC=retina_snn_evaluation_progress_report_ja"

where uplatex >nul 2>&1
if errorlevel 1 (
  echo [ERROR] uplatex was not found in PATH.
  echo Install TeX Live or MiKTeX with Japanese language support.
  exit /b 1
)

where dvipdfmx >nul 2>&1
if errorlevel 1 (
  echo [ERROR] dvipdfmx was not found in PATH.
  echo Install TeX Live or MiKTeX with Japanese language support.
  exit /b 1
)

del /q "%DOC%.aux" "%DOC%.out" "%DOC%.dvi" "%DOC%.log" "%DOC%.synctex.gz" >nul 2>&1

uplatex -kanji=utf8 -synctex=1 -interaction=nonstopmode -halt-on-error "%DOC%.tex"
if errorlevel 1 exit /b 1

uplatex -kanji=utf8 -synctex=1 -interaction=nonstopmode -halt-on-error "%DOC%.tex"
if errorlevel 1 exit /b 1

findstr /i /c:"ltjsarticle.cls" /c:"luatexja" "%DOC%.log" >nul
if not errorlevel 1 (
  echo [ERROR] LuaTeX-ja was loaded unexpectedly.
  exit /b 1
)

dvipdfmx -o "%DOC%.pdf" "%DOC%.dvi"
if errorlevel 1 exit /b 1

echo Built with upLaTeX: %DOC%.pdf
endlocal

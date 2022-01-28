@echo off
cd /d %~dp0
.\virtualenv\Scripts\activate.bat && .\virtualenv\Scripts\python.exe .\code\EISART.py

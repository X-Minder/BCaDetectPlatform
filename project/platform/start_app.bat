@echo off

:: 激活你的 venv 虚拟环境
call D:\project3\.venv\Scripts\activate.bat

:: 切换到 app.py 所在目录
cd /d D:\project3\BCaDetectPlatform-main\platform

:: 启动应用
python app.py

pause

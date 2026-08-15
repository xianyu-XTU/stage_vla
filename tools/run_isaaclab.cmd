@echo off
rem 便捷封装：python tools\run_isaaclab.py <script> [args...]
rem 用法：tools\run_isaaclab.cmd scripts\train_stare.py --num_envs 32 --max_iterations 50 --headless
setlocal
python "%~dp0run_isaaclab.py" %*
exit /b %errorlevel%

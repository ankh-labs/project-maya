@echo off
rem The ONLY tree tonight with evidence of working: cd52717 = backup-pre-20261030-0835.
rem Built in its own directory so the current build stays intact.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CL=/FS
set SCCACHE_DISABLE=1

G:\AI\maya-eval\project-maya\.venv\Scripts\cmake.exe -G Ninja -DCMAKE_MAKE_PROGRAM=G:\AI\maya-eval\project-maya\.venv\Scripts\ninja.exe -S G:\AI\maya-eval\maya-morning -B G:\AI\maya-eval\maya-morning\build-morning -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" -DSTRATA_GGML_DIR=G:\AI\maya-eval\project-maya\third_party\llama.cpp -DCMAKE_CXX_FLAGS=/FS "-DCMAKE_CUDA_FLAGS=-Xcompiler=/FS" || exit /b 3

G:\AI\maya-eval\project-maya\.venv\Scripts\cmake.exe --build G:\AI\maya-eval\maya-morning\build-morning --target maya -j 6 || exit /b 4
exit /b 0

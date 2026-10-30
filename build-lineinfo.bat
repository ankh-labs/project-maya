@echo off
rem Build with -lineinfo so compute-sanitizer can name the source line of the first bad access.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul

G:\AI\maya-eval\project-maya\.venv\Scripts\cmake.exe -G Ninja -DCMAKE_MAKE_PROGRAM=G:\AI\maya-eval\project-maya\.venv\Scripts\ninja.exe -S G:\AI\maya-eval\project-maya -B G:\AI\maya-eval\project-maya\build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" -DSTRATA_GGML_DIR=G:\AI\maya-eval\project-maya\third_party\llama.cpp "-DCMAKE_CUDA_FLAGS=-lineinfo" || exit /b 3

G:\AI\maya-eval\project-maya\.venv\Scripts\cmake.exe --build G:\AI\maya-eval\project-maya\build --target maya -j 16 || exit /b 4
exit /b 0

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# echo RUN 8GPU
# NAME=8GPU_ScaledBS
# python train_sampling_multi_gpu.py --gpu 0,1,2,3,4,5,6,7 --batch-size 512 >> "$NAME.log"
# echo RUN 4GPU
# NAME=4GPU_ScaledBS
# python train_sampling_multi_gpu.py --gpu 0,1,2,3 --batch-size 512 >> "$NAME.log"
echo RUN 2GPU
NAME=2GPU_ScaledBS
python train_sampling_multi_gpu.py --gpu 0,1 --batch-size 1024 >> "$NAME.log"
echo RUN 1GPU
NAME=1GPU_ScaledBS
python train_sampling_multi_gpu.py --gpu 0 --batch-size 512 >> "$NAME.log"

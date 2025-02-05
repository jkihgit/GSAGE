export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

TIMEOUT=36000
echo RUN 8GPU NSYS
NAME=8GPU_NSYSScaledBS_ATTMPT2
nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python train_sampling_multi_gpu.py --gpu 0,1,2,3,4,5,6,7 --batch-size 4096 >> "$NAME.log"
echo RUN 4GPU NSYS
NAME=4GPU_NSYSScaledBS_ATTMPT2
nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python train_sampling_multi_gpu.py --gpu 0,1,2,3 --batch-size 2048 >> "$NAME.log"
echo RUN 2GPU NSYS
NAME=2GPU_NSYSScaledBS_ATTMPT2
nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python train_sampling_multi_gpu.py --gpu 0,1 --batch-size 1024 >> "$NAME.log"
echo RUN 1GPU NSYS
NAME=1GPU_NSYSScaledBS_ATTMPT2
nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python train_sampling_multi_gpu.py --gpu 0 >> "$NAME.log"

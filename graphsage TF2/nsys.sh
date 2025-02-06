export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

BS=512
TIMEOUT=36000
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# for DS in pubmed ppi reddit
# do
#     echo RUN 8GPU $DS
#     NAME=8GPU_16prefetchDepth_$DS
#     nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python -m graphsage.supervised_train --train_prefix data/$DS/$DS --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --validate_batch_size $BS --sigmoid --log_dir devtest --epochs 2 --noval 1 >> "$NAME.log"
# done
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# for DS in pubmed ppi reddit
# do
#     echo RUN 4GPU $DS
#     NAME=4GPU_16prefetchDepth_$DS
#     nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python -m graphsage.supervised_train --train_prefix data/$DS/$DS --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --validate_batch_size $BS --sigmoid --log_dir devtest --epochs 2 --noval 1 >> "$NAME.log"
# done
export CUDA_VISIBLE_DEVICES=0,5
for DS in pubmed ppi reddit
do
    echo RUN 2GPU $DS
    NAME=GPU3n5_force_colocate_$DS
    nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python -m graphsage.supervised_train --train_prefix data/$DS/$DS --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --validate_batch_size $BS --sigmoid --log_dir devtest --epochs 2 --noval 1 >> "$NAME.log"
done
# export CUDA_VISIBLE_DEVICES=0
# for DS in pubmed ppi reddit
# do
#     echo RUN 1GPU $DS
#     NAME=1GPU_16prefetDepth_$DS
#     nsys profile -d $TIMEOUT -w true --sample=cpu -t 'nvtx,cuda' -o ./$NAME python -m graphsage.supervised_train --train_prefix data/$DS/$DS --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --validate_batch_size $BS --sigmoid --log_dir devtest --epochs 2 --noval 1 >> "$NAME.log"
# done
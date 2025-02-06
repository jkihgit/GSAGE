export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

BS=512
NAME="bs"$BS
# export CUDA_VISIBLE_DEVICES=7
# python3 -m graphsage.supervised_train --train_prefix data/pubmed/pubmed --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --sigmoid --log_dir devtest --epochs 100 >> PM100_1GPU.log
# export CUDA_VISIBLE_DEVICES=6,7
# python3 -m graphsage.supervised_train --train_prefix data/pubmed/pubmed --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --sigmoid --log_dir devtest --epochs 100 >> PM100_2GPU.log
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3 -m graphsage.supervised_train --train_prefix data/pubmed/pubmed --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --sigmoid --log_dir devtest --epochs 100 >> PM100_8GPU.log
# python3 -m graphsage.supervised_train --train_prefix data/reddit/reddit --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --sigmoid --log_dir devtest --epochs 100 >> R100_8GPU.log
export CUDA_VISIBLE_DEVICES=0,5
for DS in pubmed ppi reddit
# nvprof --metrics all python3 -m graphsage.supervised_train --train_prefix data/pubmed/pubmed --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size $BS --sigmoid --log_dir devtest --epochs 1 --noval 1 --minimini 1 >> R100_8GPU.log
# python -m graphsage.supervised_train --train_prefix data/reddit_8192/reddit_8192 --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 8192 --validate_batch_size 512 --sigmoid --log_dir devtest --epochs 100 --noval 1 --minimini 1 >> test.log
# 
do
    echo RUN $DS
    python -m graphsage.supervised_train --train_prefix data/$DS/$DS --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 512 --validate_batch_size 512 --sigmoid --log_dir devtest --epochs 10 >> $DS".log"

    # nsys profile -d 60 -w true --sample=cpu -t 'nvtx,cuda' -o ./R8K_minimini_randomized_"$I" python -m graphsage.supervised_train --train_prefix data/reddit_8192/reddit_8192 --model graphsage_mean --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 8192 --validate_batch_size 512 --sigmoid --log_dir devtest --epochs 100 --noval 1 --minimini 1

done
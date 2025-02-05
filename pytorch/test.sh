python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'reddit' --gpu '0' --validation 0 > R_e100_1GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'reddit' --gpu '0,1' --validation 0 > R_e100_2GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'reddit' --gpu '0,1,2,3' --validation 0 > R_e100_4GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'reddit' --gpu '0,1,2,3,4,5,6,7' --validation 0 > R_e100_8GPU.log

python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'ppi' --gpu '0' --validation 0 > PPI_e100_1GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'ppi' --gpu '0,1' --validation 0 > PPI_e100_2GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'ppi' --gpu '0,1,2,3' --validation 0 > PPI_e100_4GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'ppi' --gpu '0,1,2,3,4,5,6,7' --validation 0 > PPI_e100_8GPU.log

python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'pubmed' --gpu '0' --validation 0 > PM_e100_1GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'pubmed' --gpu '0,1' --validation 0 > PM_e100_2GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'pubmed' --gpu '0,1,2,3' --validation 0 > PM_e100_4GPU.log
python -m graphsage.model --num-epochs 100 --eval-every 9999999999 --dataset 'pubmed' --gpu '0,1,2,3,4,5,6,7' --validation 0 > PM_e100_8GPU.log
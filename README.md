# alphafold_nf_sound
#/home/sor4003/store_sor4003/folddock/SpeedPPI

https://gitlab.com/ElofssonLab/FoldDock
https://github.com/patrickbryant1/SpeedPPI
https://www.genomeweb.com/proteomics-protein-research/deepminds-alphafold-seeing-uptake-protein-protein-interaction-work#:~:text=More%20recently%2C%20researchers%20have%20begun,generating%20models%20of%20their%20structures.


######### how to run speedPPI

>>>>>>>>>>>start screen 
1# request i-session with GPU ------ take very long time 
srun --pty --partition=scu-gpu --gres=gpu:1 --mem=150G bash -i

2# module load SpeedPPI/37d0a03

3# export CUDA path 
export LD_LIBRARY_PATH=/home/software/spack/opt/spack/linux-centos7-x86_64/gcc-8.2.0/cuda-11.8.0-rqftjjg3pwtogsetgcrrytjcqutxgtaj/lib64:$LD_LIBRARY_PATH

3a### conda activate /home/sor4003/anaconda3/envs/speedPPI

4#  python3 ./src/test_gpu_avail.py
if above returns GPU --- then i loaded things efficiently 

5### (speedPPI) [sor4003@scu-node053 SpeedPPI]$ bash predict_single.sh ./data/dev/zswim8_A7E2V4.fasta ./data/dev/zswim8_A7E2V4_copy.fasta hh-suite/build/bin/hhblits 0.5 ./zswim8_zswim8/
MSAs exists...
Checking if all are present
zswim8_A7E2V4
./zswim8_zswim8//msas//zswim8_A7E2V4.a3m exists
zswim8_A7E2V4_copy
./zswim8_zswim8//msas//zswim8_A7E2V4_copy.a3m exists
Predicting...






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

5###  [sor4003@scu-node053 SpeedPPI]$ bash predict_single.sh ./data/dev/zswim8_A7E2V4.fasta ./data/dev/zswim8_A7E2V4_copy.fasta hh-suite/build/bin/hhblits 0.5 ./zswim8_zswim8/
MSAs exists...
Checking if all are present
zswim8_A7E2V4
./zswim8_zswim8//msas//zswim8_A7E2V4.a3m exists
zswim8_A7E2V4_copy
./zswim8_zswim8//msas//zswim8_A7E2V4_copy.a3m exists
Predicting...
^[[A^[[A^[[A^[[AEvaluating pair zswim8_A7E2V4-zswim8_A7E2V4_copy
2024-01-03 15:52:46.853428: W external/org_tensorflow/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc:492] The NVIDIA driver's CUDA version is 11.5 which is older than the ptxas CUDA version (11.8.89). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/mapping.py:49: FutureWarning: jax.tree_flatten is deprecated, and will be removed in a future release. Use jax.tree_util.tree_flatten instead.
  values_tree_def = jax.tree_flatten(values)[1]
/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/mapping.py:53: FutureWarning: jax.tree_unflatten is deprecated, and will be removed in a future release. Use jax.tree_util.tree_unflatten instead.
  return jax.tree_unflatten(values_tree_def, flat_axes)
/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/mapping.py:124: FutureWarning: jax.tree_flatten is deprecated, and will be removed in a future release. Use jax.tree_util.tree_flatten instead.
  flat_sizes = jax.tree_flatten(in_sizes)[0]

  ####### working with elongin B and C prediction 




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
Predicting...
Evaluating pair elongC-elongB
2024-01-03 17:01:27.844951: W external/org_tensorflow/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc:492] The NVIDIA driver's CUDA version is 11.5 which is older than the ptxas CUDA version (11.8.89). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/mapping.py:49: FutureWarning: jax.tree_flatten is deprecated, and will be removed in a future release. Use jax.tree_util.tree_flatten instead.
  values_tree_def = jax.tree_flatten(values)[1]
/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/mapping.py:53: FutureWarning: jax.tree_unflatten is deprecated, and will be removed in a future release. Use jax.tree_util.tree_unflatten instead.
  return jax.tree_unflatten(values_tree_def, flat_axes)
/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/mapping.py:124: FutureWarning: jax.tree_flatten is deprecated, and will be removed in a future release. Use jax.tree_util.tree_flatten instead.
  flat_sizes = jax.tree_flatten(in_sizes)[0]
It took 311.3439619541168 s to predict the interaction.
elongC-elongB 0.6160406819875499


######### clullin3 and zswim8 OOM
      Entry Parameter Subshape: f32[11,508,2605,49]
                ==========================

        Buffer 12:
                Size: 1.62GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/alphafold_iteration/distogram_head/add" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/modules.py" source_line=1372
                XLA Label: fusion
                Shape: f32[2605,2605,64]
                ==========================

        Buffer 13:
                Size: 559.67MiB
                Entry Parameter Subshape: f32[11,5120,2605]
                ==========================

        Buffer 14:
                Size: 559.67MiB
                Entry Parameter Subshape: s32[11,5120,2605]
                ==========================

        Buffer 15:
                Size: 559.67MiB
                Entry Parameter Subshape: f32[11,5120,2605]
                ==========================

The stack trace below excludes JAX-internal frames.
The preceding is the original exception that occurred, unmodified.

--------------------

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/./src/run_alphafold_single.py", line 258, in <module>
    main(num_ensemble=1,
  File "/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/./src/run_alphafold_single.py", line 222, in main
    prediction_result = model_runner.predict(processed_feature_dict)
  File "/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/model.py", line 133, in predict
    result = self.apply(self.params, jax.random.PRNGKey(0), feat)
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 34912723376 bytes.
BufferAssignment OOM Debugging.
BufferAssignment stats:
             parameter allocation:    5.43GiB
              constant allocation:    38.7KiB
        maybe_live_out allocation:    1.73GiB
     preallocated temp allocation:   32.51GiB
  preallocated temp fragmentation:     2.0KiB (0.00%)
                 total allocation:   39.68GiB
              total fragmentation:    1.73GiB (4.37%)
Peak buffers:
        Buffer 1:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/mul" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/modules.py" source_line=1307
                XLA Label: fusion
                Shape: f32[128,2605,2605]
                ==========================

        Buffer 2:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/mul" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/modules.py" source_line=1307
                XLA Label: fusion
                Shape: f32[128,2605,2605]
                ==========================

        Buffer 3:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/left_projection/...cb,cd->...db/jit(_einsum)/dot_general[dimension_numbers=(((0,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/common_modules.py" source_line=76
                XLA Label: custom-call
                Shape: f32[128,6786025]
                ==========================

        Buffer 4:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/left_gate/...cb,cd->...db/jit(_einsum)/dot_general[dimension_numbers=(((0,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/common_modules.py" source_line=76
                XLA Label: custom-call
                Shape: f32[128,6786025]
                ==========================

        Buffer 5:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/right_projection/...cb,cd->...db/jit(_einsum)/dot_general[dimension_numbers=(((0,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/common_modules.py" source_line=76
                XLA Label: custom-call
                Shape: f32[128,6786025]
                ==========================

        Buffer 6:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/right_gate/...cb,cd->...db/jit(_einsum)/dot_general[dimension_numbers=(((0,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/common_modules.py" source_line=76
                XLA Label: custom-call
                Shape: f32[128,6786025]
                ==========================

        Buffer 7:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/triangle_multiplication_outgoing/gating_linear/...cb,cd->...db/jit(_einsum)/dot_general[dimension_numbers=(((0,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/common_modules.py" source_line=76
                XLA Label: custom-call
                Shape: f32[128,6786025]
                ==========================

        Buffer 8:
                Size: 3.24GiB
                XLA Label: fusion
                Shape: f32[128,2605,2605]
                ==========================

        Buffer 9:
                Size: 3.24GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/broadcast_in_dim[shape=(2605, 2605, 128) broadcast_dimensions=()]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/modules.py" source_line=352
                XLA Label: broadcast
                Shape: f32[2605,2605,128]
                ==========================

        Buffer 10:
                Size: 3.18GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/while/body/alphafold_iteration/evoformer/__layer_stack_no_state/while/body/extra_msa_stack/outer_product_mean/layer_norm_input/jit(_var)/reduce_sum[axes=(2,)]" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/modules.py" source_line=1446
                XLA Label: fusion
                Shape: f32[5120,2605,64]
                ==========================

        Buffer 11:
                Size: 2.66GiB
                Entry Parameter Subshape: f32[11,508,2605,49]
                ==========================

        Buffer 12:
                Size: 1.62GiB
                Operator: op_name="jit(apply_fn)/jit(main)/alphafold/alphafold_iteration/distogram_head/add" source_file="/athena/kleavelandlab/store/sor4003/folddock/SpeedPPI/src/alphafold/model/modules.py" source_line=1372
                XLA Label: fusion
                Shape: f32[2605,2605,64]
                ==========================

        Buffer 13:
                Size: 559.67MiB
                Entry Parameter Subshape: f32[11,5120,2605]
                ==========================

        Buffer 14:
                Size: 559.67MiB
                Entry Parameter Subshape: s32[11,5120,2605]
                ==========================

        Buffer 15:
                Size: 559.67MiB
                Entry Parameter Subshape: f32[11,5120,2605]



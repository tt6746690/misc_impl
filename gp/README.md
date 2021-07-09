

```
# install cudnn
# https://developer.nvidia.com/rdp/cudnn-download
# download both runtime/dev and
sudo dpkg -i <libcudd>.deb
```


```
# os.environ['TF_CPP_VMODULE'] = '=bfc_allocator=1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# some environ set for non-default cuda install locations 
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# os.environ['LD_LIBRARY_PATH'] = '${LD_LIBRARY_PATH}:/usr/local/cuda/lib64'
```


## todos

- convgp on cub/cxr dataset 
    - performance
- p(Z) as PPS/DPP ?
    - initialize with cluster center ?
    - want to re-initialize inducing variables if not used as evidence...
- argument for using large weights of posterior mean as prototypes ?
    - decompose posterior process to 2 component 
    - data component is deterministic function of inducing variables
    - dist component is unrelated to `u` ... used to detect ood data ?

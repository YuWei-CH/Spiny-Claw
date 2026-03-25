# Setup Scripts

This directory stores one-time setup scripts that prepare the local and remote environment before the optimization loop starts.

Current scripts:

- [`scripts/init_env.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_env.sh)
  Creates the `fi-bench` conda environment and installs `flashinfer-bench` and `modal`.
- [`scripts/init_cuda_toolchain.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_cuda_toolchain.sh)
  Checks whether `nvcc`, CUDA headers, and CUDA development libraries are available locally for future `local_compile` use.
- [`scripts/init_dataset.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_dataset.sh)
  Installs `git-lfs`, clones the benchmark dataset, and prints the `FIB_DATASET_PATH` export command.
- [`scripts/init_modal_volume.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_modal_volume.sh)
  Uploads the dataset to the Modal volume and lists the volume contents.

Recommended local setup order:

1. Run [`scripts/init_env.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_env.sh)
2. Run [`scripts/init_cuda_toolchain.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_cuda_toolchain.sh) if you want local CUDA compile prechecks
3. Activate the `fi-bench` environment
4. Run `modal setup` locally to authenticate this machine with your Modal account
5. Run [`scripts/init_dataset.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_dataset.sh)
6. Export `FIB_DATASET_PATH`
7. Run [`scripts/init_modal_volume.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_modal_volume.sh)

Notes:

- `modal run` does not require a local GPU
- `modal run` does require local Modal authentication
- local `nvcc` is still recommended for the future `local_compile` precheck path
- CUDA toolkit installation is treated as one-time machine setup, not as part of the `local_compile` skill itself

These scripts are intended to be called manually first, then wrapped by future OpenClaw skills such as `init_modal` and `sync_benchmark_data`.

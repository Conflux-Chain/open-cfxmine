# cfxmine

A C++ miner for Conflux PoW.

For detailed mining instruction, please refer to [Conflux Tethys GPU Mining Instruction](https://forum.conflux.fun/t/topic/3775).

## Build

`cfxmine` depends on [CMake](https://cmake.org/) (version 3.18 or higher), and [Boost](https://www.boost.org/) (version 1.65.1).

On Linux, run the following command in a shell to build.

```bash
git submodule update --init
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
```

On Windows, alternatively run:

```bash
git submodule update --init
cmake -A x64 -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Run

cfxmine works with [Conflux-Rust](https://github.com/Conflux-Chain/conflux-rust) together. In order to run cfxmine, here are the steps:

1. Start Conflux-Rust with stratum enabled. In the configuration file, set
``mining_type = "stratum"``. By default, it will open 32525 port at the public address
of the client. You can also change the port in the configuration file.

2. Run ``./build/bin/cfxmine --addr A.B.C.D --port 32525 --gpu``, where ``A.B.C.D`` is the
public ip address of the client.

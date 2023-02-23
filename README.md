# Stable Baslines with JAX & Haiku

Implementation of Stable Baselines based on JAX & Haiku.

This library is based on Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3).

## Implemented Algorithms

| **Name**       | **Online_learning** | `Box`       		| `Discrete`         | `MultiDiscrete`     | `MultiBinary`      |
|----------------|---------------------| ------------------ | ------------------ | ------------------- | ------------------ |
| PPO            | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                |
| BC			 | :x:                 | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                |
| OnlineBC       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                |
| DT			 | :x:                 | :heavy_check_mark: | :x:                | :x:                 | :x:                |


### Install

```
git clone https://github.com/kwk2696/sb3-jax-haiku.git
pip install -e .
```


### Benchmark
We use Intel i9-10940X, RTX 3090 to benchmark Decision Transformer (DT) on MuJoCo Ant environment.
| **Name**       | **Torch-DT**        | **JAX-DT**      	|
|----------------|---------------------| ------------------ |
| DT             | 0.03 step / sec     | 3 step / sec       |


## Example

Example codes are available in ``tests`` directory.


## Currently Working On ...

Decision Transformer, Soft-Actor-Critic ...

# 单机多卡推理，使用huggingface框架
CUDA_VISIBLE_DEVICES=0,1,2 python dist_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-hf --num-samples=10 --batch_size=1

# 多机多卡推理，使用pipeline和原本的transformer框架（以太网口）

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node 3 --nnodes=3 --node_rank=0 --master_addr=10.10.10.11 --master_port=1234 pipeline_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-chat --num-samples=10 --batch_size=1

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node 3 --nnodes=3 --node_rank=1 --master_addr=10.10.10.11 --master_port=1234 pipeline_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-chat --num-samples=10 --batch_size=1

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 --nnodes=3 --node_rank=2 --master_addr=10.10.10.11 --master_port=1234 pipeline_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-chat --num-samples=10 --batch_size=1

# 多机多卡推理，使用pipeline和原本的transformer框架（高速IB网口）

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node 3 --nnodes=3 --node_rank=0 --master_addr=12.12.12.11 --master_port=1234 pipeline_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-chat --num-samples=10 --batch_size=1

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node 3 --nnodes=3 --node_rank=1 --master_addr=12.12.12.11 --master_port=1234 pipeline_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-chat --num-samples=10 --batch_size=1

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 --nnodes=3 --node_rank=2 --master_addr=12.12.12.11 --master_port=1234 pipeline_baseline.py --dataset ./scrambled_sampled_dataset.json --model /public/home/asc03/shujiuhe/datasets/Llama-2-70b-chat --num-samples=10 --batch_size=1

SHELL := /bin/bash
-include .env
LAB_DATA ?= $(abspath $(CURDIR)/../llmopt-data)
MODELOPT_TAG ?= 0.47.0
CUDA_RUN := cuda_12.9.1_575.57.08_linux.run

.PHONY: help setup cuda check models-llm third_party data ptq-smoke ptq ptq-report trtllm deploy profile deploy-report quantize-draft specdec specdec-report eagle3 profile-specdec clean-exports

help:
	@grep -hE '^[a-z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-15s %s\n", $$1, $$2}'

setup: ## Create/refresh .venv from uv.lock
	uv sync

cuda: ## Install CUDA 12.9.1 toolkit into $LAB_DATA/cuda-12.9
	mkdir -p $(LAB_DATA)/tmp
	cd $(LAB_DATA)/tmp && [ -f $(CUDA_RUN) ] || curl -L -o $(LAB_DATA)/tmp/$(CUDA_RUN) \
		https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/$(CUDA_RUN)
	sh $(LAB_DATA)/tmp/$(CUDA_RUN) --silent --toolkit --toolkitpath=$(LAB_DATA)/cuda-12.9 \
		--tmpdir=$(LAB_DATA)/tmp --no-man-page --override
	rm -f $(LAB_DATA)/tmp/$(CUDA_RUN)

check: ## Smoke-test the whole stack
	python scripts/check_env.py

models-llm: ## Qwen3-8B (~16.4 GB)
	bash scripts/download_models.sh llm

third_party: ## Clone NVIDIA/Model-Optimizer at the installed version
	[ -d third_party/Model-Optimizer ] || git clone --depth 1 --branch $(MODELOPT_TAG) \
		https://github.com/NVIDIA/Model-Optimizer.git third_party/Model-Optimizer

data: ## MT-Bench questions, EAGLE-3 training prompts, public EAGLE-3 head for Qwen3-8B
	python scripts/prepare_data.py

ptq-smoke: ## Qwen3-8B PTQ study smoke test: bf16 + int4_awq, 20 samples per task
	python scripts/ptq_study.py configs/ptq_qwen3_8b.yaml --formats bf16,int4_awq --limit 20

ptq: ## Qwen3-8B PTQ study: format grid, then calibration ablation
	-python scripts/ptq_study.py configs/ptq_qwen3_8b.yaml
	python scripts/ptq_study.py configs/ptq_qwen3_8b_calib.yaml

ptq-report: ## Tables + accuracy-vs-size plot from results/ptq_qwen3_8b.jsonl
	python scripts/report_ptq.py configs/ptq_qwen3_8b.yaml

trtllm: ## TensorRT-LLM 1.0 venv in $LAB_DATA/venvs/trtllm
	bash scripts/setup_trtllm.sh

deploy: ## Serve + benchmark each engine/format on real kernels
	python scripts/bench_deploy.py configs/deploy_qwen3_8b.yaml

profile: ## Nsight Systems batch-1 decode profile, TRT-LLM BF16 vs INT4-AWQ + INT4 lm_head
	bash scripts/profile_decode.sh trtllm_bf16 Qwen/Qwen3-8B
	bash scripts/profile_decode.sh trtllm_int4_awq $(LAB_DATA)/exports/qwen3-8b_int4_awq
	bash scripts/profile_decode.sh trtllm_int4_awq_lm_head $(LAB_DATA)/exports/qwen3-8b_int4_awq_lm_head 64 --patch-lm-head

deploy-report: ## Speed / memory / accuracy / profile tables + plots
	python scripts/report_deploy.py configs/deploy_qwen3_8b.yaml

quantize-draft: ## INT4 weight-only copy of the public EAGLE-3 head
	python scripts/quantize_draft.py --drafter_path $(LAB_DATA)/heads/tengyunw_qwen3_8b_eagle3 \
		--qformat int4_blockwise_weight_only --quantize_lm_head --exclude "*fc*" \
		--export_path $(LAB_DATA)/heads/tengyunw_qwen3_8b_eagle3_int4

specdec: ## EAGLE-3 targets x drafts x draft lengths on MT-Bench
	python scripts/bench_specdec.py configs/specdec_qwen3_8b.yaml

specdec-report: ## Speculative decoding table + speed-vs-draft-length plot
	python scripts/report_specdec.py configs/specdec_qwen3_8b.yaml

eagle3: ## Train our own EAGLE-3 head: self-generated data, draft vocab, training, validation, export
	N_CONV=10000 bash scripts/eagle3_pipeline.sh all

profile-specdec: ## Nsight Systems: one EAGLE-3 step split into target / draft / idle (INT4 everything, draft length 4)
	bash scripts/profile_decode.sh int4_lm_head_nospec $(LAB_DATA)/exports/qwen3-8b_int4_awq_lm_head 256 --patch-lm-head
	bash scripts/profile_decode.sh int4_lm_head_eagle3_int4_k4 $(LAB_DATA)/exports/qwen3-8b_int4_awq_lm_head 256 \
		--patch-lm-head --draft $(LAB_DATA)/heads/tengyunw_qwen3_8b_eagle3_int4 --draft-length 4
	python scripts/split_specdec_profile.py $(LAB_DATA)/profiles/int4_lm_head_eagle3_int4_k4.nsys-rep \
		--name int4_lm_head_eagle3_int4_k4

clean-exports: ## Delete exported checkpoints
	rm -rf $(LAB_DATA)/exports/*

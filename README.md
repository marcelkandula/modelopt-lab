# modelopt-lab

Quantization, TensorRT-LLM deployment and EAGLE-3 speculative decoding of **Qwen3-8B** with
**NVIDIA Model Optimizer** (ModelOpt), on a single **RTX A6000** (Ampere, 48 GB). Every optimization is measured twice:
accuracy on five benchmarks, and speed on real serving kernels.

## Results

Batch-1 decode on MT-Bench with TensorRT-LLM 1.0. Accuracy is the mean of MMLU, GSM8K, ARC-Challenge, HellaSwag and
Winogrande. 

| Configuration | Weights | Mean accuracy | Decode speed | vs BF16 |
|---|---|---|---|---|
| BF16 | 16.4 GB | 72.5% | 40.6 tok/s | 1.0x |
| INT4-AWQ + INT4 `lm_head` | 5.3 GB | 70.4% | 119.5 tok/s | 2.9x |
| + EAGLE-3 draft head (BF16) | 6.1 GB | 70.4% | 191.0 tok/s | 4.7x |
| **+ EAGLE-3 draft head quantized to INT4** | **5.6 GB** | **70.4%** | **235.6 tok/s** | **5.8x** |

![Accuracy vs size](report/figures/ptq_qwen3_8b_accuracy_size.png)


Tables and plots, generated from `results/`: [PTQ study](report/ptq_qwen3_8b.md),
[deployment](report/deploy_qwen3_8b.md), [speculative decoding](report/specdec_qwen3_8b.md).

## Experiments

Each experiment is one config, its scripts, one results file (`results/*.jsonl`, one row per measurement) and a report
generated from it.

| Experiment | Config | Scripts | Report |
|---|---|---|---|
| Accuracy of 12 PTQ configurations, calibration ablation | `ptq_qwen3_8b.yaml`, `ptq_qwen3_8b_calib.yaml` | `ptq_study.py`, `report_ptq.py` | [ptq](report/ptq_qwen3_8b.md) |
| Speed and memory on vLLM / TensorRT-LLM, accuracy through the server | `deploy_qwen3_8b.yaml` | `export_checkpoint.py`, `bench_deploy.py`, `check_export.py`, `report_deploy.py` | [deploy](report/deploy_qwen3_8b.md) |
| Decode profile, INT4 `lm_head` | `deploy_qwen3_8b.yaml` | `profile_decode.py`, `summarize_profile.py`, `bench_lm_head.py`, `trtllm_lm_head_patch.py` | [deploy](report/deploy_qwen3_8b.md) |
| EAGLE-3 speculative decoding, INT4 draft head | `specdec_qwen3_8b.yaml` | `prepare_data.py`, `specdec_run.py`, `bench_specdec.py`, `quantize_draft.py`, `report_specdec.py` | [specdec](report/specdec_qwen3_8b.md) |
| Training an EAGLE-3 head | `eagle3_qwen3.yaml`, `eagle3_data.yaml` | `eagle3_pipeline.sh` | [specdec](report/specdec_qwen3_8b.md) |
| Where a speculative decoding step goes | – | `profile_decode.py --draft`, `split_specdec_profile.py` | [specdec](report/specdec_qwen3_8b.md) |


## Reproduce

```bash
source env.sh
make setup
make cuda
source env.sh
make check
make models-llm
make third_party
make data
make trtllm
make ptq && make ptq-report
make deploy && make profile && make deploy-report
make quantize-draft
make eagle3
make specdec && make specdec-report
make profile-specdec
```

## License

[MIT](LICENSE)

python evaluate/eval_gate_thresholds.py \
    --data-output-dir /data1/finetune/data/wearableai_val \
    --output-dir  ~/workspace/proactive_vlm/experiments/2026-07-02/evaluate_threshold \
    --lora-adapter /data1/finetune/model/gate_lora_adapter \
    --thresholds 0.05:0.95:0.05 \
    --top-k 1
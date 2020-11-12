EXP_DIR="exp_configs/pve"; \
STYLE_FLAGS=""; \
REL_METHOD="PUCB_EBH"; \
PROCESSES=8; \
OUTPUT_DIR="results/pve"; \
mkdir -p $OUTPUT_DIR && \
scripts/run_configs.sh $EXP_DIR $PROCESSES $OUTPUT_DIR && \
scripts/plot_configs.sh $OUTPUT_DIR $OUTPUT_DIR $REL_METHOD $STYLE_FLAGS

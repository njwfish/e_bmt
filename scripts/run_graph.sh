EXP_DIR="exp_configs/supp/graph"; \
STYLE_FLAGS="--log_scale"; \
REL_METHOD="e-BH"; \
PROCESSES=8; \
OUTPUT_DIR="results/supp/graph"; \
mkdir -p $OUTPUT_DIR && \
scripts/run_configs.sh $EXP_DIR $PROCESSES $OUTPUT_DIR && \
scripts/plot_configs.sh $OUTPUT_DIR $OUTPUT_DIR $REL_METHOD $STYLE_FLAGS

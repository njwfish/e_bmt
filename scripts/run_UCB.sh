EXP_DIR="exp_configs/supp/UCB" \
STYLE_FLAGS="--style_map styles/supp.json" \
REL_METHOD="PM-H (E)"; \
PROCESSES=8; \
OUTPUT_DIR="results/supp/UCB"; \
mkdir -p $OUTPUT_DIR && \
scripts/run_configs.sh $EXP_DIR $PROCESSES $OUTPUT_DIR && \
scripts/plot_configs.sh $OUTPUT_DIR $OUTPUT_DIR "${REL_METHOD}" "${STYLE_FLAGS}"

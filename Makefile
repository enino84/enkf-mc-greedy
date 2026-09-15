SCALE ?= smoke
IMAGE ?= qgloc:latest
SHARDS ?= 4
SHARD_INDEX ?= 0
SHARD_COUNT ?= 1
DOCKER_RUN = docker run --rm -e SCALE=$(SCALE) -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 \
	-v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper $(IMAGE)

.PHONY: rescore build test smoke quick paper all cache single bench figures shards shell clean clean-cache local-test local-all local-figures pdf

build:          ; docker build -t $(IMAGE) .
test:           ; $(DOCKER_RUN) python -m pytest tests -q
smoke:          ; $(MAKE) all SCALE=smoke
quick:          ; $(MAKE) all SCALE=quick
paper:          ; $(MAKE) all SCALE=paper
all:            ; $(DOCKER_RUN) bash /work/scripts/run_all.sh $(SCALE)
cache:          ; $(DOCKER_RUN) python -c "import sys; sys.path.insert(0,'/work/experiments'); from common import *; make_testbed(get_scale('$(SCALE)'))"
single:         ; $(DOCKER_RUN) python /work/experiments/exp01_single_cycle.py $(SCALE)
bench:          ; $(DOCKER_RUN) python /work/experiments/exp02_benchmark.py $(SCALE)
figures:        ; $(DOCKER_RUN) bash /work/scripts/figures.sh $(SCALE)
exp05:          ; docker run -d --name qgloc-exp05 -e SCALE=$(SCALE) -e SHARD_INDEX=$(SHARD_INDEX) -e SHARD_COUNT=$(SHARD_COUNT) -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 -v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper $(IMAGE) python /work/experiments/exp05_first_analysis.py $(SCALE)
exp03:          ; docker run -d --name qgloc-exp03 -e SCALE=$(SCALE) -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 -v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper $(IMAGE) python /work/experiments/exp03_flow_probe.py $(SCALE)
# EXP-02 split over $(SHARDS) background containers (compute the cache first)
shards:         ; $(MAKE) cache; for i in $$(seq 0 $$(( $(SHARDS) - 1 ))); do \
	docker run -d --name qgloc-shard$$i -e SCALE=$(SCALE) -e SHARD_INDEX=$$i -e SHARD_COUNT=$(SHARDS) \
	-e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 -v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper \
	$(IMAGE) python /work/experiments/exp02_benchmark.py $(SCALE); done
shell:          ; docker run --rm -it -v $(PWD):/work -v $(PWD)/results:/work/results $(IMAGE) bash
clean:          ; rm -rf results/EXP-* results/figures_* results/*.log results/FINDINGS-BENCH_*.md paper/tables_*.tex
clean-cache:    ; rm -rf results/cache
# without docker
local-test:     ; PYTHONPATH=$(PWD) python3 -m pytest tests -q
local-all:      ; PYTHONPATH=$(PWD):$(PWD)/experiments MPLBACKEND=Agg SCALE=$(SCALE) bash -c 'cd $(PWD) && python3 experiments/exp01_single_cycle.py $(SCALE) && python3 experiments/exp02_benchmark.py $(SCALE) && $(MAKE) local-figures SCALE=$(SCALE)'
local-figures:  ; PYTHONPATH=$(PWD):$(PWD)/experiments MPLBACKEND=Agg python3 figures/fig_benchmark.py $(SCALE) && PYTHONPATH=$(PWD):$(PWD)/experiments python3 figures/fig_assignment.py $(SCALE) && PYTHONPATH=$(PWD):$(PWD)/experiments python3 figures/make_tables.py $(SCALE)
pdf:            ; cd paper && pdflatex -interaction=nonstopmode assignment.tex >/dev/null && pdflatex -interaction=nonstopmode assignment.tex | grep -E "Output|Error" ; rm -f paper/*.aux paper/*.log paper/*.out
# re-score a finished benchmark with another cycle window, e.g. make rescore SCALE=paper BURN=20
BURN ?= 15
rescore:        ; $(DOCKER_RUN) python /work/figures/rescore.py $(SCALE) --burn-in $(BURN) && $(MAKE) figures SCALE=$(SCALE)
# EXP-04: steady-state probe; pass options in PROBE, e.g. make exp04 SCALE=paper PROBE="--network random-moving --N 80"
PROBE ?=
exp04:          ; docker run -d --name qgloc-exp04 -e SCALE=$(SCALE) -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 -v $(PWD)/results:/work/results -v $(PWD)/paper:/work/paper $(IMAGE) python /work/experiments/exp04_probe.py $(SCALE) $(PROBE)

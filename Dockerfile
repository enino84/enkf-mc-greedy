# syntax=docker/dockerfile:1
# qgloc — adaptive localization by observation assignment on the QG testbed.
#
#   docker compose build
#   SCALE=paper docker compose up -d suite        # whole suite, one container
#   docker compose logs -f suite
#
# results/ and paper/ are bind-mounted: everything written survives the
# container, and results/cache keeps the climatology so a second run starts
# in seconds.
FROM python:3.12-slim

# Deterministic numerics: one BLAS thread and a fixed hash seed, otherwise the
# runs are reproducible only up to thread scheduling.
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg RESULTS_DIR=/work/results \
    PYTHONPATH=/work:/work/experiments

# pyteda pulls pyshtools and netCDF4, which need a compiler and these libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev libfftw3-dev \
        libhdf5-dev libnetcdf-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /work/requirements.txt

COPY qgloc /work/qgloc
COPY experiments /work/experiments
COPY figures /work/figures
COPY scripts /work/scripts
COPY tests /work/tests
COPY paper/assignment.tex paper/nota_criterio.tex paper/nota_bayes_alpha.tex /work/paper/
COPY README.md EXPERIMENTS.md FINDINGS.md Makefile setup.py /work/

RUN mkdir -p /work/results/cache && chmod +x /work/scripts/*.sh \
    && python -m pytest tests -q

CMD ["bash", "/work/scripts/run_all.sh"]

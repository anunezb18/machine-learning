# Makefile — Challenge 7, Group 8 (SVHN → MNIST)

PYTHON  = python3
DATA    = ./data
SYNTH   = ./data/synthetic_target
FIGS    = ./figures
CKPTS   = ./checkpoints

.PHONY: all install part_a part_b part_c fast clean

all: part_a part_b part_c

install:
	pip install -r requirements.txt

# Full runs (3 seeds each)
part_a:
	$(PYTHON) main.py --parts A --data_root $(DATA) --figures_dir $(FIGS) --ckpt_dir $(CKPTS)

part_b:
	$(PYTHON) main.py --parts B --data_root $(DATA) --synth_dir $(SYNTH) --figures_dir $(FIGS)

part_c:
	$(PYTHON) main.py --parts C --data_root $(DATA) --synth_dir $(SYNTH) \
	                  --figures_dir $(FIGS) --ckpt_dir $(CKPTS)

# Quick smoke-test (1 seed, fast mode)
fast:
	$(PYTHON) main.py --parts ABC --fast --no_dann \
	                  --data_root $(DATA) --synth_dir $(SYNTH) \
	                  --figures_dir $(FIGS) --ckpt_dir $(CKPTS)

clean:
	rm -rf $(FIGS) $(CKPTS) $(SYNTH)
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

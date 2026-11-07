# LLM Proposes, Governance Disposes — IEEE WCNC paper (LaTeX)

## Layout
```
main.tex              6-page IEEEtran conference paper (pdfLaTeX, Overleaf-ready)
references.bib        references
IEEEtran.cls/.bst     IEEE class + style (Overleaf already provides them)
figures/scenario.tex  Fig. 1 scenario (TikZ)
figures/arch.tex      Fig. 2 architecture (TikZ)
figures/workflow.tex  Fig. 3 governed tool-call workflow (TikZ)
tables/live.tex       Table I, generated from the PoC's MongoDB results (do not edit by hand)
tables/live.json      every metric behind Table I
```

## What the evaluation uses
* **ColO-RAN** (Polese et al., IEEE TMC 2023; GPL-3.0): real srsRAN stack on the Colosseum emulator,
  7 BSs, 3 slices, 3 schedulers x 28 RBG splits. The PoC's `scripts/prepare_coloran.py` turns the
  per-UE reports into one-second slice records (724,086), which the PoC consumes as they are.
* **Live LLM runs only**: the complete PoC (Monitoring agent, Near-RT and Non-RT RIC agents, governance
  with its LLM review) with NVIDIA Nemotron 3.5 Lightning through a LiteLLM gateway. Every score is
  computed from the stored records by deterministic rules (`scripts/agent_checks.py` in the PoC).
  - R1 = runs `claude-run2`, `claude-run3`, `claude-run4` (call budget 20)
  - R2 = run `claude-run5` (call budget 2, compact prompts)
  - R0 = runs `new-run`, `claude-run` (previous revision, quoted in the text only)

## Regenerate the tables (from the PoC root)
```
poetry run python scripts/paper_tables.py \
  --group "R1=claude-run2,claude-run3,claude-run4" --group "R2=claude-run5" \
  --caption "Live LLM agents behind the gate (Nemotron~3.5 Lightning; R1: no call cap, R2: two calls, compact prompts)" \
  --tex <paper>/tables/live.tex
poetry run python scripts/stability.py stab-1 ... stab-10 --name nemotron --tex <paper>/tables/stability.tex
poetry run python scripts/compare_models.py <run ids of every model> --name paper \
  --labels nvidia-network-slicing-model=Nemotron,mistral-network-slicing-model=Ministral-8B \
  --tex <paper>/tables/models.tex
```
`main.tex` includes the stability and model tables only once those files exist.

The dataset statistics of Sec. V-A (violation rates per slice, PRB-limited and persistent shares) are
computed by the PoC's SLA analyzer on `data/coloran_prepared_dataset.csv`.

## Earlier material not used by the current paper
`data/`, `scripts/` and `figures/fig_*.pdf` hold the experiments of an earlier version (trace-driven
closed loop, fault-injection benchmark, memory scaling), which used scripted stand-ins for the LLM
agents and the previous SLA definitions. They are kept for reference only.

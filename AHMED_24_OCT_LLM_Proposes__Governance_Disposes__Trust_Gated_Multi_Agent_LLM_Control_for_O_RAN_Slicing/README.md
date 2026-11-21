# LLM Proposes, Governance Disposes — IEEE WCNC paper (LaTeX)

## Layout
```
main.tex                    6-page IEEEtran conference paper (pdfLaTeX, Overleaf-ready)
references.bib              references
IEEEtran.cls/.bst           IEEE class + style (Overleaf already provides them)
figures/scenario.tex        Fig. 1 scenario (TikZ)
figures/arch.tex            Fig. 2 architecture (TikZ)
figures/workflow.tex        Fig. 3 governed tool-call workflow (TikZ)
figures/live_gov.tex        governance outcomes of the live runs (generated, do not edit)
figures/gate_latency.tex    gate latency (generated, do not edit)
tables/livenewvf.tex        Table I, live LLM agents behind the gate
tables/live.json            every metric behind Table I and figures/live_gov.tex
main.pdf                    compiled paper
```

Build: `latexmk -pdf main.tex` (build files are git-ignored).

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

## Regenerate the generated files (from the PoC root)
```
poetry run python scripts/paper_tables.py \
  --group "R1=claude-run2,claude-run3,claude-run4" --group "R2=claude-run5" \
  --caption "..." --tex <paper>/tables/live.tex
poetry run python scripts/paper_figures.py --metrics <paper>/tables/live.json \
  --tex <paper>/figures/live_gov.tex \
  --latency results/gate_latency.json --latency-live results/gate_latency_llm.json \
  --latency-tex <paper>/figures/gate_latency.tex
```
`tables/livenewvf.tex` is the edited version of the generated table that `main.tex` uses.

The dataset statistics of Sec. V-A (violation rates per slice, PRB-limited and persistent shares) are
computed by the PoC's SLA analyzer on `data/coloran_prepared_dataset.csv`.

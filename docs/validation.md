# Validation record

Checked on 26 October 2026 using the versions in `requirements.txt` and Python 3.13.

## Original execution

All 13 original Python files parsed. All four optimisation scripts and the
`finalopt.solve_task2` helper solved successfully. Both standalone exploratory
scripts executed with the locally provided dataset, with interactive-display and
library-deprecation messages in the initial headless run.

The combined dashboard failed at startup with `ModuleNotFoundError: finalopt`.
All three classifiers failed in reduced-search diagnostic runs because pandas 3
left their replacement-based target labels with an object type that current
scikit-learn did not recognise as a classification target. The logistic script
also referenced `results_df` without defining it.

## Corrected execution

- All four optimisation scripts, both standalone exploratory scripts and all three
  classifiers completed through `run_checks.py --quick`.
- The corrected batch logs contained no warnings or tracebacks.
- Model checks used all supplied rows with reduced parameter searches. The original
  exhaustive parameter searches were not completed and their optimal parameters
  or original performance have not been claimed as reproduced.
- Five dashboard section paths executed in Streamlit AppTest with the local input.
  The custom option-menu return value was mocked to select each section.
- The scheduling button returned an optimal skills-constrained solution costing
  £1,784, and the scheduling page started without reading the patient input.
- Synthetic tests covered data sections and the raw-data Apply filters action,
  including pandas 3 string columns being available for categorical filtering.
- Ten automated tests pass. Streamlit emits its normal bare-testing-context
  messages in the test harness; these are distinct from app exceptions.
- Exported schedules were checked for 14 daily hours, 70 weekly hours, availability,
  minimum weekly commitments and the applicable skill coverage constraints.
- The 13 original code-file hashes remain unchanged.

## Changes

The portable copy uses `machine_learning/` and `optimisation/` directories, a root
Streamlit entry point and configurable local data paths. Draft optimisation filenames
were given descriptive names. Group coursework attribution is preserved.

The corrected label mapping explicitly produces integer targets. Logistic feature
selection now operates on the same transformed feature order used for downstream
training and interpretation. Its validation trade-off table is constructed before
plotting. Threshold selection has a defined fallback when no value on the original
grid meets the requested recall, rather than returning `None`.

Mixed numeric/string summary columns are prepared for display without Arrow
serialization fallback errors. Deprecated container-width and boxplot options were
updated, categorical filtering recognises current pandas string dtypes, and the
age-group plotting boundary includes age zero. Identifiers are omitted from generic
plot-variable menus. The skills-scheduling interface label now matches its backend.

The optimisation helper checks solver status after its first optimisation stage
before reading solution values. Original scheduling inputs and objectives were retained.

## Not verified

The custom navigation component's frontend, every browser interaction, exhaustive
model tuning, prospective prediction validity, cross-site generalisation, raw-data
editing/deletion edge cases, concurrent-user persistence and alternate solver licence
configurations were not exhaustively tested. Data and clinical interpretation limits
are stated in the README.

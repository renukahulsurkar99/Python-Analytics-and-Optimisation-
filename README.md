# Python Analytics and Optimisation

Group coursework combining integer-programming models for staff scheduling with
exploratory analysis, breach classification and an interactive accident and
emergency department data dashboard.

**Attribution:** BMAN73701 Group K coursework, presented in
[Renuka Hulsurkar's portfolio](https://github.com/renukahulsurkar99). This is a group
project and is not presented as solely authored work. Individual team contributions
have not been specified in the supplied material.

## Project components

| Component | Techniques |
| --- | --- |
| Staff scheduling | Integer programming in Gurobi, availability and weekly commitments |
| Cost and fairness | Minimum-cost, minimax workload and lexicographic range/cost scenarios |
| Skill coverage | Daily programming and troubleshooting coverage constraints |
| AED exploratory analysis | Descriptive statistics, distributions, correlations and categorical tests |
| Breach classifiers | Logistic Regression, Decision Tree and Random Forest |
| Classification evaluation | Stratified train/validation/test split, class weighting, ROC-AUC and recall-constrained threshold selection |
| Dashboard | Streamlit navigation, filtering, session-scoped record operations and activity logging |

The scheduling case and patient-data analysis are separate coursework problems
combined in one interface. The scheduling inputs are embedded case-study values,
not actual patient data or a claim about hospital staffing requirements.

## Scheduling results

The [exported schedules and scenario comparison](outputs/optimisation/) can be
inspected without the AED dataset. Every scenario assigns 70 hours across five
days and respects operator availability and minimum weekly commitments.

| Scenario | Objective | Observed cost (£) | Maximum weekly operator hours | Weekly hours range |
| --- | --- | ---: | ---: | ---: |
| Base schedule | Minimise cost | 1,755 | 20 | 13 |
| Minimax schedule | Minimise maximum weekly hours within the cost cap | 1,784 | 14 | 6 |
| Range schedule | Minimise workload range, then cost | 1,784 | 14 | 6 |
| Skills scenario | Apply skill coverage, minimise range, then cost | 1,784 | 14 | 6 |

The minimax model has no secondary cost objective. Its displayed cost is the
cost of the returned optimal minimax solution, not a separate minimum-cost claim.
The combined dashboard's solve button runs the final skills-constrained scenario;
the batch exporter compares all four.

## Setup

Python 3.13 was used for validation. Use Python 3.12 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

Gurobi needs a working solver licence. The small coursework models ran with the
restricted non-production licence available in the test installation. No solver
licence file or account credential is included in this repository.

Run the scheduling cases and export results:

```bash
python run_optimisation.py
```

Start the dashboard:

```bash
streamlit run app.py
```

The scheduling page works without patient data. Other dashboard sections require
the locally supplied CSV described in [the data guide](data/README.md).

## Run the data analyses and classifiers

With the local CSV configured:

```bash
python run_checks.py --quick
```

This executes all four optimisation cases, two exploratory scripts and all three
classifiers. Quick mode uses the complete supplied dataset with reduced tuning:
one parameter combination per model, three-fold cross-validation, 30 Random Forest
trees and a larger RFECV elimination step. It verifies execution, not an exhaustive
hyperparameter search or a final model-performance claim.

To run the retained original search grids, omit `--quick`. This can take much
longer. The full search grids were not rerun during the publication check.

For just the four optimisation scripts:

```bash
python run_checks.py --optimisation-only
```

Patient-derived figures, aggregate model summaries, console logs and dashboard
activity logs are saved locally and excluded from Git. Source patient records,
identifiers and row-level prediction outputs are not included in the public
repository.

## Validation and runtime fixes

```bash
python -m unittest discover -s tests -v
```

Ten tests cover recall-threshold edge cases, scheduling without a patient input,
the solve button and dashboard sections using synthetic records. The tests mock
the third-party navigation component's selected value, while executing the actual
page code and native Streamlit widgets. This does not test browser styling or the
custom component's JavaScript.

Publication fixes include portable data/module paths, integer class labels for
pandas 3, construction of the previously undefined logistic trade-off table,
consistent feature ordering during feature selection, a feasible threshold fallback,
headless plot saving, mixed-type table display and a solver-status guard.
The supplied source folder was not edited.

See [validation notes](docs/validation.md) for the checked scope and remaining limits.

## Modelling limitations

- The four-week patient sample has a strongly imbalanced target. Recall alone
  does not describe false-positive burden; precision and ROC-AUC are also reported
  locally. A validation recall target does not guarantee the same recall on test data.
- Classifier checks used reduced tuning. They should not be cited as the results
  of the original exhaustive searches.
- The scripts retain random stratified splits, rather than a chronological or
  externally validated study design. No prospective or cross-site evaluation was
  performed.
- The availability time of investigation/treatment counts and other fields has
  not been established. The model should not be described as an arrival-time
  predictor without checking whether those inputs would already be known.
- Preprocessing is fitted on the training partition before internal cross-validation.
  A stricter model-selection design would fit preprocessing inside each fold.
- The dashboard's raw-data operations affect session copies and local activity
  logs. They do not update the source CSV. Editing, deletion and concurrent-user
  behaviour were not exhaustively tested.
- This is coursework demonstrating programming and analysis, not a validated
  system for making patient-care decisions.

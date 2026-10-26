# Local AED input

Supply `AED4weeks.csv` in this directory. Alternatively set the environment variable
`AED_DATA_PATH` to its absolute path. Source records are excluded from version control.

The code expects these columns:

| Field | Expected content |
| --- | --- |
| `ID` | Record identifier, excluded from classifier predictors |
| `Age` | Numeric age |
| `Day` | Numeric day index |
| `DayofWeek` | Monday–Sunday labels |
| `Period` | Numeric arrival-hour category |
| `LoS` | Length of stay in minutes, excluded from classifier predictors |
| `Breachornot` | `breach` or `non-breach` |
| `HRG` | Categorical health-related group |
| `noofinvestigation` | Numeric investigation count |
| `nooftreatment` | Numeric treatment count |
| `noofpatients` | Numeric occupancy measure |

The supplied local file contains 8,781 records with no missing values and no
duplicate IDs. Most exploratory views use a reproducible 400-record sample;
the classifiers use the full dataset and a 70/15/15 stratified split, subject to
integer rounding.

No raw records, patient-level plots, filtered downloads, audit logs or licence
files are distributed here. Model outputs generated while running the scripts
are written to ignored local folders. The public scheduling results are generated
from the embedded coursework scheduling case, independently of the AED file.

The source provider and collection context were not supplied. The repository
does not claim that the data are synthetic or approved for public redistribution.

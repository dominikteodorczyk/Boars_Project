# Boars_Project

This repository contains the source code and tools used in the **Boars_Project**, conducted at **Wrocław University of Environmental and Life Sciences (UPWr).**

## Description

The project focuses on analyzing wild boar migration patterns, studying their population.
It includes scripts for data parsing, calculations, and analysis related to wild boars.

## Directory Structure

- `constants/` - Contains predefined constants used in the project.
- `src/` - Main source code of the application.

## Key Files

- `.gitignore` – Specifies files and directories ignored by Git.
- `README.md` – This file.
- `data_parsing.py` – A script for parsing input data.
- `infostop_calculation.py` – A script for calculating points where animals stop.
- `laws_params.py` – Calculation of the law of animal mobility.
- `requirements.txt` – List of required dependencies.

## Getting Started
### How to get our solutions
1. Clone the repository:<br>
```bash
git clone https://github.com/dominikteodorczyk/Boars_Project.git`
```
2. Navigate to the project directory:<br>
```bash
cd Boars_Project`
```
3. Install dependencies:<br>
```bash
pip install -r requirements.txt`
```
### Fundamentals
Here you will get information on how to process your data and obtain results on the parameters of the mobility laws of the studied animals

#### Parse your data
First of all, prepare your data. For this, in the `cols.json` file
The JSON configuration file must have the following structure:
```json
{
    "path/to/csv_file.csv": [
        "time_column",
        "agent_id_column",
        "longitude_column",
        "latitude_column"
    ]
}
```

Each key is the path to a CSV file, and its value is a list specifying the
columns to retrieve. The list must include:
- The time column (as the first entry).
- The agent ID column (as the second entry).
- The longitude and latitude columns (as the third and fourth entries).

Also add to the `periods.json` file for breeding periods, e.g., or others in which animals can move in a different way:
```json
{
    "Boar.csv":[11,1],
}
```
Now you can enable the `data_parsing.py` script

```bash
python data_parsing.py
```

IMPORTANT!<br>
If you want to generate many complicated periods for one animal, use such a script several times on paresed data without using `periods.json`:
```python
import os
import pandas as pd
path = "Boars.csv"
df = pd.read_csv(path, parse_dates=["datetime"])
months = [10,11,12] ## enter the months for the period here
df_filtered = df[df["datetime"].dt.month.isin(months)]
df_filtered.to_csv(f"parsed_{months[0]}_{months[-1]}_{os.path.basename(path})", index=False)
```
If you have done the parsing we can go to the next step

#### Calculate stopping points using the Infostop algorithm.
To do this, use the `infostop_calculations.py` script in which you first specify in the `PARSED_DATA_DIR` variable the folder where you store the parsed data in .csv format and `OUTPUT_DIR_NAME`.  Run the script with the command:
```bash
python infostop_calculation.py
```


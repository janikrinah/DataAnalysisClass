# parse_data.py
import pandas as pd

def load_finaldata(
    owid_path: str = "owid-energy-data.csv",
    income_path: str = "income_classification.xlsx",
) -> pd.DataFrame:
    """
    Load, clean and merge the OWID energy data with the World Bank
    income classification. Returns finaldata for year 2021.
    """

    # 1) OWID energy data
    df0 = pd.read_csv(owid_path)

    selected_cols = [
        "country", "year", "iso_code",
        "population", "electricity_demand",
        "greenhouse_gas_emissions",
        "fossil_share_elec", "renewables_share_elec"
    ]

    energydata = df0[selected_cols]
    energydata = energydata[energydata["year"] == 2021]
    energydata = energydata.dropna()

    energydata = energydata.rename(columns={
        "country": "Country",
        "year": "Year",
        "iso_code": "ISO",
        "population": "Population",
        "electricity_demand": "Electricity demand",
        "greenhouse_gas_emissions": "GHG emissions",
        "fossil_share_elec": "FF electricity share",
        "renewables_share_elec": "RE electricity share",
    })

    # 2) Income classification data
    xls = pd.ExcelFile(income_path)
    df1 = pd.read_excel(xls, sheet_name="Country Analytical History")

    FY23_col       = 36
    startrow       = 11
    isocodecol     = 0
    countrynamecol = 1

    income_2021 = df1.iloc[startrow:, [isocodecol, countrynamecol, FY23_col]].copy()
    income_2021.columns = ["ISO", "Country Name", "Income Group FY23"]
    income_2021 = income_2021.dropna()

    # 3) Merge
    energydata["ISO"]  = energydata["ISO"].astype(str)
    income_2021["ISO"] = income_2021["ISO"].astype(str)

    finaldata = energydata.merge(
        income_2021[["ISO", "Income Group FY23"]],
        on="ISO",
        how="left",
    )
    finaldata = finaldata.dropna()

    return finaldata

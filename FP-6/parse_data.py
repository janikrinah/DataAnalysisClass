import pandas as pd

def load_data():
    # Load OWID energy dataset
    df0 = pd.read_csv('owid-energy-data.csv')

    selected_cols = [
        "country", "year", "iso_code",
        "population", "electricity_demand",
        "greenhouse_gas_emissions",
        "fossil_share_elec", "renewables_share_elec"
    ]

    energydata = df0[selected_cols]
    energydata = energydata[energydata['year'] == 2021]
    energydata = energydata[energydata["iso_code"].notna()]

energydata = energydata.dropna(subset=[
    "population", "electricity_demand", "greenhouse_gas_emissions",
    "fossil_share_elec", "renewables_share_elec"
])

    energydata = energydata.rename(columns={
        'country': 'Country',
        'year': 'Year',
        'iso_code': 'ISO',
        'population': 'Population',
        'electricity_demand': 'Electricity demand',
        'greenhouse_gas_emissions': 'GHG emissions',
        'fossil_share_elec': 'FF electricity share',
        'renewables_share_elec': 'RE electricity share'
    })

    # Income classification
    data = pd.ExcelFile('income_classification.xlsx')
    df1 = pd.read_excel(data, sheet_name='Country Analytical History')

    FY23_col = 36
    startrow = 11
    income_2021 = df1.iloc[startrow:, [0, 1, FY23_col]].dropna()
    income_2021.columns = ["ISO", "Country Name", "Income Group FY23"]

    energydata["ISO"]  = energydata["ISO"].astype(str)
    income_2021["ISO"] = income_2021["ISO"].astype(str)

    finaldata = energydata.merge(
        income_2021[["ISO", "Income Group FY23"]],
        on="ISO",
        how="left"
    ).dropna()

    return finaldata
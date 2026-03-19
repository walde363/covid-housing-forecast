import pandas as pd
from pygris.data import get_lodes
from pygris import counties

# 1. Download County Geometries (for the key)
print("Downloading Florida County geometries to map the names...")
fl_counties = counties(state="FL", year=2020)
# GEOID is the 5-digit county FIPS, NAMELSAD is the plain name (e.g. "Miami-Dade County")
county_key = fl_counties[['GEOID', 'NAMELSAD']]

# 2. Iterate through Years 2016-2023
all_years_data = []

for year in range(2016, 2024):
    print(f"\nProcessing LODES data for Florida, Year: {year}...")
    try:
        fl_wac = get_lodes(state="FL", year=year, lodes_type="wac", cache=True)
        fl_rac = get_lodes(state="FL", year=year, lodes_type="rac", cache=True)

        # The first 5 characters of a Block GEOID represent the County.
        fl_wac['county_geoid'] = fl_wac['w_geocode'].astype(str).str[:5]
        fl_rac['county_geoid'] = fl_rac['h_geocode'].astype(str).str[:5]

        # Aggregate Jobs vs. Residents by County
        # C000 is the column for 'Total Jobs'
        jobs_per_county = fl_wac.groupby('county_geoid')['C000'].sum().reset_index(name='total_jobs')
        residents_per_county = fl_rac.groupby('county_geoid')['C000'].sum().reset_index(name='resident_workers')

        # Calculate Commute-Shed Pressure Ratio
        pressure_df = pd.merge(jobs_per_county, residents_per_county, on='county_geoid', how='inner')
        pressure_df['pressure_ratio'] = pressure_df['total_jobs'] / pressure_df['resident_workers']
        pressure_df['year'] = year
        
        all_years_data.append(pressure_df)
        print(f"Successfully processed {year}.")
        
    except Exception as e:
        print(f"Error processing {year}: {e}")

# 3. Combine and Merge Names
if all_years_data:
    final_df = pd.concat(all_years_data, ignore_index=True)

    # Merge key onto all data
    final_df_with_names = final_df.merge(
        county_key, 
        left_on='county_geoid', 
        right_on='GEOID', 
        how='left'
    )

    # Cleanup Columns into a neat order
    final_df_with_names.rename(columns={'NAMELSAD': 'county_name'}, inplace=True)
    final_df_with_names = final_df_with_names[['year', 'county_geoid', 'county_name', 'total_jobs', 'resident_workers', 'pressure_ratio']]

    print("\nPreview of Multi-Year County Commute Data:")
    print(final_df_with_names.head())

    # Save for the model
    final_df_with_names.to_csv('fl_county_commute_pressure_2016_2023.csv', index=False)
    print("\nSaved output to fl_county_commute_pressure_2016_2023.csv")
else:
    print("\nNo data to save.")
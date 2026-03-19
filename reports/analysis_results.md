# Data Quality Report

## File: `processed_data.csv`

- **Rows**: 67
- **Columns**: 41
- **Duplicates**: 0

### Missing Values

|                              |   Missing Count |   Percentage (%) |
|:-----------------------------|----------------:|-----------------:|
| highest_category             |              61 |         91.0448  |
| FLUR                         |               1 |          1.49254 |
| Investor Purchases_FL        |              45 |         67.1642  |
| Investor Market Share_FL     |              45 |         67.1642  |
| Investor Purchases YoY_FL    |              45 |         67.1642  |
| Investor Purchases_Total     |              45 |         67.1642  |
| Investor Market Share_Total  |              45 |         67.1642  |
| Investor Purchases YoY_Total |              45 |         67.1642  |

### Data Types

| Column                                     | Data Type   |
|:-------------------------------------------|:------------|
| date                                       | object      |
| highest_category                           | float64     |
| FLUR                                       | float64     |
| FLBP1FH                                    | int64       |
| Investor Purchases_FL                      | float64     |
| Investor Market Share_FL                   | float64     |
| Investor Purchases YoY_FL                  | float64     |
| Investor Purchases_Total                   | float64     |
| Investor Market Share_Total                | float64     |
| Investor Purchases YoY_Total               | float64     |
| HPI_South_Atlantic_Not_Seasonally_Adjusted | float64     |
| HPI_South_Atlantic_Seasonally_Adjusted     | float64     |
| HPI_USA_Not_Seasonally_Adjusted            | float64     |
| HPI_USA_Seasonally_Adjusted                | float64     |
| Inventory_FL                               | float64     |
| Inventory_Non_FL                           | float64     |
| Inventory_US                               | float64     |
| DaysToClose_FL                             | float64     |
| DaysToClose_Non_FL                         | float64     |
| DaysToClose_US                             | float64     |
| MarketTemp_FL                              | float64     |
| MarketTemp_Non_FL                          | float64     |
| MarketTemp_US                              | float64     |
| NewConSales_FL                             | float64     |
| NewConSales_Non_FL                         | float64     |
| NewConSales_US                             | float64     |
| ZHVI_Tier_FL                               | float64     |
| ZHVI_Tier_Non_FL                           | float64     |
| ZHVI_Tier_US                               | float64     |
| ZORDI_Condo_FL                             | float64     |
| ZORDI_Condo_Non_FL                         | float64     |
| ZORDI_Condo_US                             | float64     |
| ZORDI_MFR_FL                               | float64     |
| ZORDI_MFR_Non_FL                           | float64     |
| ZORDI_MFR_US                               | float64     |
| ZORDI_SFR_FL                               | float64     |
| ZORDI_SFR_Non_FL                           | float64     |
| ZORDI_SFR_US                               | float64     |
| ZORDI_All_FL                               | float64     |
| ZORDI_All_Non_FL                           | float64     |
| ZORDI_All_US                               | float64     |

### Summary Statistics (Numerical)

|                                            |   count |         mean |         std |          min |         25% |          50% |          75% |         max |
|:-------------------------------------------|--------:|-------------:|------------:|-------------:|------------:|-------------:|-------------:|------------:|
| highest_category                           |       6 |      2.66667 |     1.21106 |      1       |      2      |      2.5     |      3.75    |      4      |
| FLUR                                       |      66 |      4.08485 |     1.8194  |      2.8     |      3.1    |      3.5     |      4.2     |     11.6    |
| FLBP1FH                                    |      67 |  10546.5     |  1637.04    |   6975       |   9413.5    |  10626       |  11737       |  14372      |
| Investor Purchases_FL                      |      22 |  13077.4     |  4368.45    |   8670       |   9968.5    |  11469       |  14881.2     |  21969      |
| Investor Market Share_FL                   |      22 |     22.9621  |     3.07315 |     15.8333  |     21.2083 |     22.5833  |     25.4583  |     27.5    |
| Investor Purchases YoY_FL                  |      22 |      5.18939 |    48.3299  |    -45.6667  |    -16.7083 |    -10.3333  |      7.83333 |    161.5    |
| Investor Purchases_Total                   |      22 |  62275.7     | 18040       |  44926       |  49898.8    |  53429       |  69635.5     |  99412      |
| Investor Market Share_Total                |      22 |     15.7921  |     1.85089 |     10.8409  |     14.8769 |     15.9848  |     17.1487  |     18.7386 |
| Investor Purchases YoY_Total               |      22 |      6.67984 |    38.0564  |    -39.7992  |    -12.1771 |      1.13636 |      6.7732  |    132.216  |
| HPI_South_Atlantic_Not_Seasonally_Adjusted |      67 |    413.041   |    55.9404  |    294.08    |    372.08   |    427       |    463.24    |    472.1    |
| HPI_South_Atlantic_Seasonally_Adjusted     |      67 |    409.089   |    55.294   |    288.85    |    370.175  |    423.17    |    455.42    |    466.99   |
| HPI_USA_Not_Seasonally_Adjusted            |      67 |    389.076   |    46.1058  |    289.47    |    357.48   |    396.9     |    430.295   |    443.75   |
| HPI_USA_Seasonally_Adjusted                |      67 |    385.792   |    45.6004  |    283.54    |    356.385  |    392.19    |    423.455   |    440.36   |
| Inventory_FL                               |      67 |   2515.7     |   840.374   |   1251.59    |   1838.23   |   2437.4     |   3241.75    |   4083.12   |
| Inventory_Non_FL                           |      67 |    600.677   |   120.679   |    374.386   |    508.074  |    589.511   |    692.585   |    835.928  |
| Inventory_US                               |      67 |    660.824   |   136.878   |    401.977   |    560.89   |    646.296   |    758.913   |    927.681  |
| DaysToClose_FL                             |      67 |     35.0697  |     2.01587 |     31.6471  |     33.5397 |     34.6471  |     36.4412  |     39.419  |
| DaysToClose_Non_FL                         |      67 |     35.1393  |     2.23858 |     32.4305  |     33.5176 |     34.2486  |     37.2716  |     40.3995 |
| DaysToClose_US                             |      67 |     35.1301  |     2.19097 |     32.3405  |     33.5566 |     34.2804  |     37.1071  |     40.2072 |
| MarketTemp_FL                              |      67 |     50.1523  |    16.3677  |     31.3793  |     34.686  |     44.4483  |     64.8621  |     81.4828 |
| MarketTemp_Non_FL                          |      67 |     58.9746  |    12.5287  |     37.3539  |     48.3817 |     59.8202  |     66.4679  |     85.2957 |
| MarketTemp_US                              |      67 |     58.6974  |    12.6003  |     37.1939  |     48.0195 |     59.2198  |     66.4034  |     85.1094 |
| NewConSales_FL                             |      67 |    339.771   |    56.6491  |    207.783   |    313.5    |    349.652   |    372.804   |    455.522  |
| NewConSales_Non_FL                         |      67 |    364.444   |    72.8209  |    212.558   |    312.219  |    359.852   |    423.657   |    503.655  |
| NewConSales_US                             |      67 |    362.501   |    71.0222  |    212.651   |    310.509  |    358.621   |    418.156   |    499.798  |
| ZHVI_Tier_FL                               |      67 | 337112       | 43705.3     | 238698       | 309777      | 357779       | 369323       | 373678      |
| ZHVI_Tier_Non_FL                           |      67 | 264333       | 24429.5     | 205659       | 250542      | 272977       | 283539       | 290629      |
| ZHVI_Tier_US                               |      67 | 266692       | 24977       | 206729       | 252461      | 276056       | 286202       | 292720      |
| ZORDI_Condo_FL                             |      67 |     51.5423  |    39.0814  |     10.5909  |     19.1136 |     39.1     |     75.8     |    137.389  |
| ZORDI_Condo_Non_FL                         |      67 |     75.9988  |    39.5362  |     27.6931  |     42.779  |     66.75    |    104.992   |    167.957  |
| ZORDI_Condo_US                             |      67 |     73.3809  |    38.8775  |     26.4754  |     40.9328 |     63.3463  |    101.381   |    163.881  |
| ZORDI_MFR_FL                               |      67 |     93.5684  |    66.5271  |      6.53846 |     43.8    |     76.125   |    135.531   |    231.636  |
| ZORDI_MFR_Non_FL                           |      67 |     92.0337  |    44.9429  |     14.9212  |     60.2141 |     82.3563  |    122.243   |    195.324  |
| ZORDI_MFR_US                               |      67 |     92.2147  |    45.8299  |     14.6146  |     59.542  |     81.8246  |    123.625   |    197.332  |
| ZORDI_SFR_FL                               |      67 |     90.4968  |    69.8767  |     17.963   |     35.3929 |     62.68    |    127.093   |    284.708  |
| ZORDI_SFR_Non_FL                           |      67 |    120.709   |    64.6941  |     38.7858  |     71.1594 |     95.7806  |    168.157   |    284.902  |
| ZORDI_SFR_US                               |      67 |    119.239   |    64.5768  |     38.0178  |     70.1823 |     94.0894  |    166.85    |    283.842  |
| ZORDI_All_FL                               |      67 |     84.7374  |    62.0916  |     15.2069  |     33.9286 |     66.5926  |    127.109   |    242.833  |
| ZORDI_All_Non_FL                           |      67 |     98.3704  |    47.2461  |     29.4371  |     61.2914 |     86.1141  |    136.413   |    208.986  |
| ZORDI_All_US                               |      67 |     97.8753  |    47.6965  |     28.9633  |     60.2685 |     85.3901  |    135.648   |    210.488  |

### Summary Statistics (Categorical)

|      |   count |   unique | top        |   freq |
|:-----|--------:|---------:|:-----------|-------:|
| date |      67 |       67 | 2020-06-01 |      1 |

## File: `processed_data_low.csv`

- **Rows**: 62828
- **Columns**: 15
- **Duplicates**: 0

### Missing Values

|             |   Missing Count |   Percentage (%) |
|:------------|----------------:|-----------------:|
| MarketTemp  |             746 |          1.18737 |
| NewConSales |           42133 |         67.0609  |
| ZHVI_Tier   |            2037 |          3.24219 |
| ZORDI_Condo |           48043 |         76.4675  |
| ZORDI_MFR   |           25107 |         39.9615  |
| ZORDI_SFR   |           24769 |         39.4235  |
| ZORDI_All   |           15385 |         24.4875  |
| DaysToClose |           53757 |         85.5622  |

### Data Types

| Column      | Data Type   |
|:------------|:------------|
| RegionID    | int64       |
| SizeRank    | int64       |
| RegionName  | object      |
| RegionType  | object      |
| StateName   | object      |
| Month       | object      |
| MarketTemp  | float64     |
| date        | object      |
| NewConSales | float64     |
| ZHVI_Tier   | float64     |
| ZORDI_Condo | float64     |
| ZORDI_MFR   | float64     |
| ZORDI_SFR   | float64     |
| ZORDI_All   | float64     |
| DaysToClose | float64     |

### Summary Statistics (Numerical)

|             |   count |        mean |          std |       min |    25% |      50% |      75% |             max |
|:------------|--------:|------------:|-------------:|----------:|-------:|---------:|---------:|----------------:|
| RegionID    |   62828 | 423713      | 101482       | 394297    | 394555 | 394811   | 395070   | 845172          |
| SizeRank    |   62828 |    467.297  |    270.245   |      1    |    233 |    467   |    700   |    939          |
| MarketTemp  |   62082 |     58.3554 |     21.6794  |    -37    |     45 |     56   |     70   |    240          |
| NewConSales |   20695 |    169.023  |    306.162   |      5    |     25 |     54   |    166.5 |   3348          |
| ZHVI_Tier   |   60791 | 266940      | 165378       |  47274.2  | 166145 | 218789   | 312310   |      1.6385e+06 |
| ZORDI_Condo |   14785 |     60.644  |     66.8623  |    -48    |     22 |     40   |     73   |   1229          |
| ZORDI_MFR   |   37721 |     82.1376 |     82.6913  |    -15    |     31 |     58   |    103   |    934          |
| ZORDI_SFR   |   38059 |    107.112  |    106.036   |     -4    |     43 |     76   |    131   |   1598          |
| ZORDI_All   |   47443 |     89.8037 |     84.1123  |     -8    |     38 |     65   |    112   |   1099          |
| DaysToClose |    9071 |     34.9653 |      6.85568 |     14.25 |     31 |     33.8 |     37.5 |     71.2        |

### Summary Statistics (Categorical)

|            |   count |   unique | top          |   freq |
|:-----------|--------:|---------:|:-------------|-------:|
| RegionName |   62828 |      933 | Winfield, KS |     68 |
| RegionType |   62828 |        1 | msa          |  62828 |
| StateName  |   62828 |       51 | TX           |   4774 |
| Month      |   62828 |       68 | 2023-12-31   |    932 |
| date       |   62828 |       68 | 2023-12-01   |    932 |


from gpbasics import global_parameters as global_param

global_param.ensure_init()

import pandas as pd
import numpy as np

from main.ExperimentFramework import matching_region_analysis

y_col = ["Open", "Close", "High", "Low"]

# Exemplary dataset covers german DAX performance index in a daily resolution.
# Exemplary dataset can be retrieved from:
# Verizon Media and Deutsche Boerse, DAX Performance-Index - Yahoo Finance. https://finance.yahoo.com/quote/%5EN225/history.
df_data = pd.read_csv("data/DaxDaily_1987-2020.csv", error_bad_lines=False).dropna()

# timestamp values are converted to a decimal format.
x_data = np.array(pd.to_datetime(df_data["Date"])).astype("float")

y_data = np.reshape(np.array(np.mean(df_data[y_col[2:4]], axis=1)), [-1, 1])

# Analyze relative change instead of absolute change
y_data = y_data[1:] / y_data[:-1]

x_data = x_data[1:]

# window size corresponding to half a year
win_size = 180

# stride corresponding two about two weeks
stride = 10

# provide minimum and maximum decimalized timestamp on a by-year basis for denormalization process.
# => this makes further manual inspection and illustration of the resulting data more easy.
decimal_year_min = 1987 + (364 / 365)

decimal_year_max = 2020 + (274 / 365)

name = "Dax"

path = ""

matching_region_analysis(
    x_data, y_data, win_size, stride, decimal_year_min, decimal_year_max, name, path, ['01/01/1995', '06/30/1995'])
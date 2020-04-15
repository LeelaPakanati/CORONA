#! /usr/bin/python
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

data = pd.read_csv(sys.stdin)
df_len = data.shape[0]

data = data.divide(data.sum(axis=1), axis=0)

plt.stackplot(range(1, df_len+1), data["Infected"], data["Susceptible"], data["Recovered"], data["Deceased"], labels = ["Infected", "Susceptible", "Recovered", "Deceased"])

plt.legend(loc = "upper left")

plt.title("Virus SIRD track")

plt.show()

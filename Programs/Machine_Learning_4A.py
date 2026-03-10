# utf-8
"""
Programme de regression linéaire simple en Python
"""

# Auteur: Enzo C

import os
from tkinter import Y 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# slope = pente (a1) intercept = ordonnée à l'origine (a0)
def predict(a1, a0, x):
    return a1 * x + a0

def ComputeSimpleLinRegresCoefAnalytical(x, y):
    n = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x**2)
    sum_xy = np.sum(x * y)

    a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    a0 = (sum_y - a1 * sum_x) / n

    return a1, a0

# SOUS FORME DE MATRICE, on peut aussi calculer les coefficients a0 et a1 de la regression linéaire simple
# on cherche a0 et a1 de la forme (a1, a0) = (X^T * X)^(-1) * X^T * Y)
def ComputeSimpleLinRegresCoefMatrix(x, y):
    n = len(x)
    X = np.vstack((x, np.ones(n))).T
    Y = y.reshape(-1, 1)
    # Calcul des coefficients a1 et a0
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
    a1, a0 = coefficients.flatten()
    return a1, a0

def main():

    #Lire un fichier 
    df = pd.read_csv("CSV/rls_donnees_simple.csv", sep=",", comment="#", header = 0)

    #recuperer les valeurs de x et y
    x = df["x0"].to_numpy()
    y = df["y"].to_numpy()
    print(x)
    #print(y)

    #a1, a0 = ComputeSimpleLinRegresCoefAnalytical(x,y)
    

if __name__ == "__main__":
    main()
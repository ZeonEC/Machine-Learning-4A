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
    
    # np.vstack((np.ones(n),x)).T permet de créer une matrice X avec une colonne de 
    # 1 pour l'ordonnée à l'origine et une colonne de x pour la pente
    # np.vstack emplie des lignes en fonction de list 
    X = np.vstack((np.ones(n),x)).T
    print("Matrice X :\n", X)

    # reshape(-1, 1) permet de transformer le vecteur y en une matrice colonne
    # -1 indique que le nombre de lignes est automatiquement déterminé en fonction du nombre de colonnes (1 dans ce cas)
    # 1  indique que la matrice doit avoir une seule colonne
    Y = y.reshape(-1, 1)
    print("Matrice Y :\n", Y)

    # Calcul des coefficients a1 et a0
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
    print("Coefficients (a1, a0) :\n", coefficients)

    a1, a0 = coefficients.flatten()
    return a1, a0

def main():

    #Lire un fichier 
    df = pd.read_csv("CSV/rls_donnees_simple.csv", sep=",", comment="#", header = 0)

    #recuperer les valeurs de x et y
    x = df["x0"].to_numpy()
    y = df["y"].to_numpy()
    
    # calculer les coefficients de la regression linéaire simple
    a1, a0 = ComputeSimpleLinRegresCoefAnalytical(x, y)

    # Calculer les coefficients de la regression linéaire simple avec la méthode matricielle
    a1, a0 = ComputeSimpleLinRegresCoefMatrix(x, y)
    

if __name__ == "__main__":
    main()
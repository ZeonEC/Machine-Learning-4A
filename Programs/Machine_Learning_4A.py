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

    return a0, a1

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
    print("Coefficients (a0, a1) :\n", coefficients)

    a0, a1 = coefficients.flatten()
    return a0, a1

def ComputeSimpleLinRegresCoefGradientDescent(x, y, learning_rate=0.01, max_iterations=1000, tolerance = 1e-6):
    n = len(x)
    a0, a1 = 0.0, 0.0
    for iteration in range(max_iterations):
        y_pred = predict(a1, a0, x)
        error = y_pred - y
        
        # Calcul des gradients
        da1 = (2/n) * np.sum(error * x)
        da0 = (2/n) * np.sum(error)

        if iteration > 0 and (abs(da1 - da1_prev) < tolerance and abs(da0 - da0_prev) < tolerance):
            print(f"Convergence atteinte après {iteration} itérations.")
            break

        da1_prev = da1
        da0_prev = da0

        # Mise à jour des coefficients
        a1 -= learning_rate * da1
        a0 -= learning_rate * da0

        print("Coefficients (a0, a1) avec Gradient Descent :\n", a0, a1)
    return da0, da1

#def ComputeSimpleLinRegresCoefStochasticGradientDescentMatrix(x, y, learning_rate=0.01, max_iterations=1000, tolerance = 1e-6):

def main():

    #Lire un fichier 
    df = pd.read_csv("CSV/rls_donnees_simple.csv", sep=",", comment="#", header = 0)

    #recuperer les valeurs de x et y
    x = df["x0"].to_numpy()
    y = df["y"].to_numpy()
    
    # calculer les coefficients de la regression linéaire simple
    a0, a1 = ComputeSimpleLinRegresCoefAnalytical(x, y)

    # Calculer les coefficients de la regression linéaire simple avec la méthode matricielle
    a0, a1 = ComputeSimpleLinRegresCoefMatrix(x, y)

    # marker = "+" pour des points en +
    # "ob" pour des points en o de couleur bleue
    # plt.plot(x,y, "+b" ) pour des points en + de couleur bleue

    #traçage du nuage de points
    plt.plot(x,y, "ob" )

    # a, b = np.polyfit(x, y, 1) calcule la pente et l'ordonnée à l'origine pour vous.
    # plt.scatter(x, y) #nuage de point simple

    plt.plot(x, a1*x + a0, color='red')

    plt.ylabel('Regression linéaire')
    plt.show()

    a0, a1 = ComputeSimpleLinRegresCoefGradientDescent(x, y)
    
    

if __name__ == "__main__":
    main()
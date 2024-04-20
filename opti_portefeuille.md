# projet_python

#pip install pandas yfinance numpy matplotlib seaborn scipy pyfolio quantstats

import pandas as pd  # Pour la manipulation et l'analyse des données.
import yfinance as yf  # Utilisé pour récupérer les données financières de Yahoo Finance.
import numpy as np  # Bibliothèque pour le calcul scientifique et la manipulation de tableaux.
import matplotlib.pyplot as plt  # Pour la création de graphiques statiques, animés et interactifs.
import seaborn as sns  # Pour de belles visualisations de données, basées sur matplotlib.

from scipy.optimize import minimize  

# Pour les algorithmes d'optimisation minimale.
from scipy.optimize import differential_evolution  # Pour les algorithmes d'optimisation par évolution différentielle.
import pyfolio as pf  # Pour créer des feuilles de calcul de rendement et analyser les performances des stratégies de trading.
import quantstats as qs  # Pour les analyses de performance et de risque des séries temporelles financières.



# Présenter les tickers français par défaut
print("Tickers français par défaut: 'AIR', 'OR', 'AI', 'BNP', 'MC', 'SU', 'SAN', 'KER'")
user_input = input("Appuyez sur Entrée pour utiliser les valeurs par défaut ou entrez vos propres tickers que vous voulez optimiser: séparé d'une virgule ")


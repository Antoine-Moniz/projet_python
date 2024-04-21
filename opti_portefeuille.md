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

# Utiliser les valeurs par défaut si l'utilisateur appuie simplement sur Entrée
tickers = user_input.strip() if user_input.strip() != "" else "AIR, SU, SAN, EL, TTE, DG, BN"
print("Tickers sélectionnés:", tickers)

# Demandez à l'utilisateur de saisir une date de début, avec une date par défaut.
date_debut = input("Entrez la date de début (format YYYY-MM-DD, par défaut 2014-01-01) Appuyez sur Entrée pour utiliser les valeurs par défaut : ") or "2014-01-01"


prix = yf.download(tickers,start=date_debut)['Adj Close']  

rendements = prix.pct_change().dropna(axis=1,how='all').iloc[1:].dropna(axis=1) #calculs des rendements et supressions des lignes vides apres calculs


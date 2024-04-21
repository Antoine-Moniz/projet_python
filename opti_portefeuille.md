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

def sortino(rendements,rendement_taux_sans_risque):
    # Filtrer les rendements négatifs pour se concentrer uniquement sur le risque à la baisse. 
    #rendements[rendements<0]` crée un sous-ensemble du DataFrame qui ne contient que les valeurs négatives (les rendements à la baisse).
    val_négative=rendements[rendements<0]
    # Calculer l'écart-type des rendements négatifs, ce qui représente la volatilité du risque à la baisse.
    # Cet écart-type est annualisé en le multipliant par la racine carrée de 252, le nombre typique de jours de cotation dans une année.
    écart_type = val_négative.std() * np.sqrt(252)
    # Calculer le rendement annuel moyen composé des investissements.
    # `np.prod(rendements+1)` calcule le produit cumulatif des rendements (ajustés de 1 pour chaque période),ce qui donne la croissance totale du portefeuille sur la période.
    # Élever ce produit à la puissance de `252/len(rendements)` annualise ce rendement, Soustraire 1 pour convertir ce facteur de croissance en rendement net.
    rendement_annuel = np.product(rendements+1)**(252/len(rendements))-1
    return ( rendement_annuel - rendement_taux_sans_risque )/ écart_type  # Calculer du ratio de Sortino.

def sharpe(rendements,rendement_taux_sans_risque):
    écart_type=rendements.std()*np.sqrt(252)  # Calculer l'écart-type des rendements, ce qui représente la volatilité du risque
    rendement_annuel = np.prod(rendements+1)**(252/len(rendements))-1
    return ( rendement_annuel - rendement_taux_sans_risque )/ écart_type  # Calculer du ratio de Sharpe.


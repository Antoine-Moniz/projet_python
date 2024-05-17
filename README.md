
#pip install pandas yfinance numpy matplotlib seaborn scipy pyfolio quantstats

import pandas as pd  # Pour la manipulation et l'analyse des données.
import yfinance as yf  # Utilisé pour récupérer les données financières de Yahoo Finance.*
import numpy as np  # Bibliothèque pour le calcul scientifique et la manipulation de tableaux.
import matplotlib.pyplot as plt  # Pour la création de graphiques statiques, animés et interactifs.
import seaborn as sns  # Pour de belles visualisations de données, basées sur matplotlib.
from scipy.optimize import minimize  
# Pour les algorithmes d'optimisation minimale
from scipy.optimize import differential_evolution  # Pour les algorithmes d'optimisation par évolution différentielle.
import pyfolio as pf  # Pour créer des feuilles de calcul de rendement et analyser les performances des stratégies de trading.
* import quantstats as qs  # Pour les analyses de performance et de risque des séries temporelles financières.

## Présenter les tickers français par défaut
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
    #rendements[rendements<0] crée un sous-ensemble du DataFrame qui ne contient que les valeurs négatives (les rendements à la baisse).
    val_négative=rendements[rendements<0]
    # Calculer l'écart-type des rendements négatifs, ce qui représente la volatilité du risque à la baisse.
    # Cet écart-type est annualisé en le multipliant par la racine carrée de 252, le nombre typique de jours de cotation dans une année.
    écart_type = val_négative.std() * np.sqrt(252)
    # Calculer le rendement annuel moyen composé des investissements.
    # np.prod(rendements+1) calcule le produit cumulatif des rendements (ajustés de 1 pour chaque période),ce qui donne la croissance totale du portefeuille sur la période.
    # Élever ce produit à la puissance de 252/len(rendements) annualise ce rendement, Soustraire 1 pour convertir ce facteur de croissance en rendement net.
    rendement_annuel = np.product(rendements+1)**(252/len(rendements))-1
    return ( rendement_annuel - rendement_taux_sans_risque )/ écart_type  # Calculer du ratio de Sortino.

def sharpe(rendements,rendement_taux_sans_risque):
    écart_type=rendements.std()*np.sqrt(252)  # Calculer l'écart-type des rendements, ce qui représente la volatilité du risque
    rendement_annuel = np.prod(rendements+1)**(252/len(rendements))-1
    return ( rendement_annuel - rendement_taux_sans_risque )/ écart_type  # Calculer du ratio de Sharpe.

def calmar(rendements,rendement_taux_sans_risque):
    val_portefeuille=np.cumprod(rendements+1) #Calcule la valeur cumulée du portefeuille en supposant que tous les rendements sont réinvestis.
    peak=val_portefeuille.expanding(1).max()  #Détermine la valeur maximale du portefeuille jusqu'à chaque point dans le temps, ce qui aide à identifier les pics avant les drawdowns.
    rendement_annuel = np.prod(rendements+1)**(252/len(rendements))-1
    drawdown=(val_portefeuille-peak)/peak   #Calcule le drawdown comme la baisse relative depuis le dernier pic.
    return ( rendement_annuel - rendement_taux_sans_risque )/-np.min(drawdown)

def scipy_func(x,arguments):   # technique d'optimisation la plus rapide mais la moin fiable  =Poids_dans_le_portfeuille
    Poids=x.reshape(-1,1)     # Reformate le vecteur des poids x en une matrice colonne pour faciliter les opérations matricielles.
    ratio, rendement_taux_sans_risque, rendements = arguments   #ration -> les fonctions , rendement_taux_sans_risque -> rentabilité sans risques, rendements   -> retunrs= rentabilité de chaque actif
    val_portefeuille_rendements=pd.Series(np.dot(rendements,Poids).reshape(-1,))   # calul de la rentabilité du portefeuille
    résultat=-ratio(val_portefeuille_rendements,rendement_taux_sans_risque)   # résultat de la maximisation grace au - car on peut que minimiser avec pandas
    if np.isnan(résultat) or np.isinf(résultat): # pour enlevé les problemes de calculs si denominateur de raproche de 0 ou inf
        return 10
    return résultat


def scipy_func_avec_pénalité(x,arguments):  #algo plus complexe mais plus lent  # Définition de la fonction avec deux arguments: x (les poids des actifs) et arguments (un tuple contenant le ratio de performance, le taux de rendement sans risque et les rendements historiques des actifs).
    Poids=x.reshape(-1,1)   # Reformate le vecteur des poids x en une matrice colonne pour faciliter les opérations matricielles.
    ratio, rendement_taux_sans_risque, rendements = arguments    # Décompose le tuple arguments pour extraire le ratio de performance, le taux de rendement sans risque et les rendements historiques des actifs.
    val_portefeuille_rendements=pd.Series(np.dot(rendements,Poids).reshape(-1,))   # Calcule les rendements du portefeuille en multipliant la matrice des rendements historiques par les poids, puis transforme le résultat en une série pandas.
    résultat=-ratio(val_portefeuille_rendements,rendement_taux_sans_risque)# Applique la fonction de ratio de performance (par exemple, Sharpe, Sortino, Calmar) aux rendements du portefeuille, et multiplie le résultat par -1 car l'optimiseur cherche à minimiser cette fonction.
    pénalité=100* np.abs((np.sum(np.abs(x))-1)) # Calcule une pénalité proportionnelle à l'écart entre la somme des poids absolus des actifs et 1. Cette pénalité est ajoutée pour forcer les poids à s'additionner à 1.
    if np.isnan(résultat) or np.isinf(résultat): # Vérifie si le résultat est NaN (non défini) ou infini, ce qui peut arriver si les calculs ne sont pas valides (par exemple, division par zéro).
        return 1000 + pénalité  # Si le résultat est NaN ou infini, retourne une valeur très élevée (1000 dans ce cas) plus la pénalité, ce qui décourage l'optimiseur de choisir cette solution.
    return résultat + pénalité  # Retourne le résultat de la fonction de ratio de performance ajusté par la pénalité. Cela permet à l'optimiseur de prendre en compte à la fois la performance du portefeuille et le respect de la contrainte des poids.


def contrainte(x):  # Définit une fonction de contrainte pour l'optimisation.
    return np.sum(x) - 1  # La somme des poids des actifs doit être égale à 1. Cette ligne calcule cette somme et soustrait 1, visant à obtenir zéro 


def optimisation_Poids(x, arguments):  # Fonction d'optimisation qui utilise l'algorithme 'SLSQP' pour trouver les poids optimaux des actifs.
    cons = {'type': 'eq', 'fun': contrainte}  # Crée un dictionnaire représentant une contrainte d'égalité ('eq') qui utilise la fonction 'contrainte' définie précédemment.
    nombre_actifs = x.size  # Détermine le nombre d'actifs en examinant la taille du vecteur des poids initiaux x.
    bounds = [(0, 1) for _ in range(nombre_actifs)]  # Établit les bornes pour chaque poids d'actif, les limitant à l'intervalle [0, 1] pour chaque actif.
    résultat = minimize(scipy_func, x, args=arguments, constraints=cons, method='SLSQP')  # Appelle la fonction 'minimize' de SciPy avec la méthode 'SLSQP', en passant la fonction d'objectif 'scipy_func', le vecteur de poids initial 'x', les arguments supplémentaires, les contraintes et spécifie la méthode d'optimisation.
    return résultat  # Retourne l'objet résultat de l'optimisation, qui contient les poids optimisés parmi d'autres informations.


def optimisation_Poids_efficace(x, arguments):  # Fonction d'optimisation alternative utilisant l'algorithme 'differential_evolution'.
    nombre_actifs = len(x[0])  # Détermine le nombre d'actifs en examinant la longueur de la première sous-liste de x, qui contient les poids initiaux.
    bounds = [(-1, 1) for _ in range(nombre_actifs)]  # Établit des bornes pour chaque poids d'actif, permettant ici des valeurs négatives, ce qui peut indiquer une vente à découvert.
    def objective_function(Poids):  # Fonction d'objectif interne pour 'differential_evolution'.
        return scipy_func_avec_pénalité(Poids, arguments)  # Appelle la fonction 'scipy_func_avec_pénalité' avec les poids actuels et les arguments passés.
    résultat = differential_evolution(objective_function, bounds)  # Appelle la fonction 'differential_evolution' de SciPy avec la fonction d'objectif et les bornes. Cette méthode est plus robuste mais potentiellement plus lente que 'SLSQP'.
    return résultat  # Retourne l'objet résultat de l'optimisation, contenant les poids optimisés.
# Demander à l'utilisateur quel ratio utiliser pour l'optimisation du portefeuille, avec "Sharpe" comme valeur par défaut.
choix_ratio = input("Voulez-vous optimiser votre portefeuille selon le ratio de Sharpe, Sortino, ou Calmar? Entrez 'Sharpe', 'Sortino' ou 'Calmar' (par défaut : Sharpe) : ") or "Sharpe"
print(f"Vous avez choisi d'optimiser selon le ratio de {choix_ratio}.")

# Demander à l'utilisateur de choisir entre une méthode d'optimisation longue et très efficace ou une courte mais moins efficace, avec "courte mais moins efficace" comme valeur par défaut.
choix_optimisation = input("Voulez-vous utiliser une méthode d'optimisation 'longue et très efficace' ou 'courte mais moins efficace'? (par défaut : courte mais moins efficace) : ") or "courte mais moins efficace"

print(f"Vous avez choisi d'utiliser une méthode d'optimisation {choix_optimisation }.")
# Demander à l'utilisateur de saisir le taux sans risque, avec 0.3 comme valeur par défaut.
try:
    rendement_taux_sans_risque = float(input("Entrez le rendement du taux sans risque (en %, par défaut 0.3) : ") or 0.3)
except ValueError:
    print("Entrée invalide. Utilisation de la valeur par défaut : 0.3%")
    rendement_taux_sans_risque = 0.3

# Demander à l'utilisateur de saisir la taille de la fenêtre, avec 1000 comme valeur par défaut.
try:
    fenêtre = int(input("Entrez la taille de la fenêtre (par défaut 1000) : ") or 1000)
except ValueError:
    print("Entrée invalide. Utilisation de la valeur par défaut : 1000")
    fenêtre = 1000

Poids = [np.ones(shape=(len(rendements.columns),)) / len(rendements.columns)]  # Initialisation des poids de chaque actif dans le portefeuille de manière équitable.
x0 = Poids[0]  # Le vecteur initial de poids pour l'optimisation.

opti_Poids = []  # Initialisation de la liste pour stocker les poids optimisés à différents points dans le temps.


if choix_optimisation == "courte mais moins efficace":
    if choix_ratio == "Sharpe":
        # Boucle pour effectuer une optimisation itérative sur une fenêtre glissante des données de rendements.
        for i in range(fenêtre, len(rendements)):
            # À chaque itération, la fonction 'optimisation_Poids' est appelée avec les paramètres actuels pour optimiser les poids.
            # La fenêtre des données de rendements utilisée pour l'optimisation se déplace à chaque itération.
            poids_optimisés = optimisation_Poids(x0, [sharpe, rendement_taux_sans_risque, rendements.iloc[i-fenêtre:i]]).x
            opti_Poids.append(list(poids_optimisés))  # Les poids optimisés sont ajoutés à la liste 'opti_Poids'.
        # on a pris en compte le fait de ne pas utiliser les informations du futur.
    elif choix_ratio == "Sortino":
        for i in range(fenêtre, len(rendements)):
            poids_optimisés = optimisation_Poids(x0, [sortino, rendement_taux_sans_risque, rendements.iloc[i-fenêtre:i]]).x
            opti_Poids.append(list(poids_optimisés))
    elif choix_ratio == "Calmar":
        for i in range(fenêtre, len(rendements)):
            poids_optimisés = optimisation_Poids(x0, [calmar, rendement_taux_sans_risque, rendements.iloc[i-fenêtre:i]]).x
            opti_Poids.append(list(poids_optimisés))

else:  # longue et très efficace
    if choix_ratio == "Sharpe":
        for i in range(fenêtre, len(rendements)):
            poids_optimisés = optimisation_Poids_efficace(x0, [sharpe, rendement_taux_sans_risque, rendements.iloc[i-fenêtre:i]]).x
            opti_Poids.append(list(poids_optimisés))
    elif choix_ratio == "Sortino":
        for i in range(fenêtre, len(rendements)):
            poids_optimisés = optimisation_Poids_efficace(x0, [sortino, rendement_taux_sans_risque, rendements.iloc[i-fenêtre:i]]).x
            opti_Poids.append(list(poids_optimisés))
    elif choix_ratio == "Calmar":
        for i in range(fenêtre, len(rendements)):
            poids_optimisés = optimisation_Poids_efficace(x0, [calmar, rendement_taux_sans_risque, rendements.iloc[i-fenêtre:i]]).x
            opti_Poids.append(list(poids_optimisés))

opti_Poids=pd.DataFrame(np.array(opti_Poids), columns=rendements.columns, index =rendements.iloc[-len(np.array(opti_Poids)):].index)
# Convertir le tableau numpy des poids optimisés en un DataFrame pandas pour une manipulation plus aisée. 
# Définir les noms de colonnes pour correspondre à ceux des rendements des actifs et indexer le DataFrame avec les dates correspondantes des rendements.

backtest=(rendements.iloc[-len(np.array(opti_Poids)):]*opti_Poids).sum(axis=1)
# Calculer les rendements du backtest en multipliant les rendements des actifs par les poids optimisés correspondants pour chaque période, 
# puis en sommant ces produits pour obtenir le rendement total du portefeuille à chaque période. 
# Les données de rendement utilisées correspondent à la période couverte par les poids optimisés.

backtest[np.abs(backtest)>1]=0
# Remplacer par 0 les valeurs du backtest dont l'absolu est supérieur à 1. 
# Cela pourrait être utilisé pour éliminer les valeurs aberrantes ou les erreurs de calcul qui entraînent des rendements irréalistes.

# Demander à l'utilisateur de choisir un benchmark, avec "CAC 40" comme valeur par défaut.
choix_benchmark = input("Choisissez un benchmark donner son ticker: par default le cac40").strip().lower() or "^FCHI"




# Télécharger les données de prix ajustés de clôture pour le benchmark sélectionné à partir de l'année 2013.
cac40 = yf.download(choix_benchmark, start= date_debut)['Adj Close']




cac40_rendements=cac40.pct_change().dropna()
# Calculer les rendements quotidiens en pourcentage de l'indice S&P 500 et supprimer la première valeur qui est NaN à cause du calcul du pourcentage de changement.

achat_conservation=rendements.mean(axis=1)    
# Calculer la stratégie d'achat et de conservation (buy and hold) en prenant la moyenne des rendements de tous les actifs à chaque période.
# Cela fournit un rendement moyen quotidien pour un portefeuille équipondéré composé de tous les actifs sélectionnés.
# Trouvez la première date où les deux séries ont des données
date_debut_commune = max(backtest.dropna().index[0], cac40.dropna().index[0])

# Tronquez les séries pour qu'elles commencent à la date de début commune
backtest_tronqué = backtest[backtest.index >= date_debut_commune]
cac40_tronqué = cac40[cac40.index >= date_debut_commune]

# Convertissez les rendements quotidiens en rendements cumulatifs à partir de la date de début commune
cumulative_rendements_backtest_tronqué = (1 + backtest_tronqué).cumprod() - 1
cumulative_rendements_cac40_tronqué = (1 + cac40_tronqué.pct_change().fillna(0)).cumprod() - 1

# Tracer les rendements cumulatifs du backtest et du benchmark tronqués
plt.figure(figsize=(14, 7))
plt.plot(cumulative_rendements_backtest_tronqué, label='Backtest')
plt.plot(cumulative_rendements_cac40_tronqué, label='CAC 40')

# Ajouter des titres et des étiquettes
plt.title('Comparaison des rendements cumulés: Backtest vs Benchmark')
plt.xlabel('Date')
plt.ylabel('Rendements cumulés')
plt.legend()

# Afficher le graphique
plt.show()
import matplotlib.pyplot as plt

# Trouvez la première date où les deux séries ont des données
date_debut_commune = max(backtest.dropna().index[0], achat_conservation.dropna().index[0])

# Tronquez les séries pour qu'elles commencent à la date de début commune
backtest_tronqué = backtest[backtest.index >= date_debut_commune]
achat_conservation_tronqué = achat_conservation[achat_conservation.index >= date_debut_commune]

# Calculez les rendements cumulés pour chaque série
cumulative_rendements_backtest = (1 + backtest_tronqué).cumprod() - 1
cumulative_rendements_achat_conservation = (1 + achat_conservation_tronqué).cumprod() - 1

# Créez le graphique en utilisant matplotlib
plt.figure(figsize=(14, 7))
plt.plot(cumulative_rendements_backtest, label='Backtest')
plt.plot(cumulative_rendements_achat_conservation, label='Achat et Conservation')

# Ajoutez un titre et des étiquettes pour les axes
plt.title('Comparaison des rendements cumulés: Backtest vs Achat et Conservation')
plt.xlabel('Date')
plt.ylabel('Rendements cumulés')

# Ajoutez une légende
plt.legend()

# Affichez le graphique
plt.show()
import matplotlib.pyplot as plt

# Trouvez la première date où les deux séries ont des données
date_debut_commune = max(achat_conservation.dropna().index[0], cac40.dropna().index[0])

# Tronquez les séries pour qu'elles commencent à la date de début commune
achat_conservation_tronqué = achat_conservation[achat_conservation.index >= date_debut_commune]
cac40_tronqué = cac40[cac40.index >= date_debut_commune]

# Calculez les rendements cumulés pour chaque série
cumulative_rendements_achat_conservation = (1 + achat_conservation_tronqué).cumprod() - 1
cumulative_rendements_cac40 = (1 + cac40_tronqué.pct_change().fillna(0)).cumprod() - 1

# Créez le graphique en utilisant matplotlib
plt.figure(figsize=(14, 7))
plt.plot(cumulative_rendements_achat_conservation, label='Achat et Conservation')
plt.plot(cumulative_rendements_cac40, label='CAC 40')

# Ajoutez un titre et des étiquettes pour les axes
plt.title('Comparaison des rendements cumulés: Achat et Conservation vs CAC 40')
plt.xlabel('Date')
plt.ylabel('Rendements cumulés')

# Ajoutez une légende
plt.legend()

# Affichez le graphique
plt.show()



 

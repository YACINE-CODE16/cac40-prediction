# 📈 CAC40 Trend Prediction

Application Streamlit pour la prédiction de tendance du CAC40 utilisant des techniques de Machine Learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Fonctionnalités

- **Visualisation interactive** des cours du CAC40 avec Plotly
- **Indicateurs techniques** : RSI, MACD, Moyennes Mobiles, Bandes de Bollinger
- **Prédiction de tendance** basée sur les indicateurs techniques
- **Analyse statistique** des rendements
- **Interface moderne** avec thème dark mode

## 🚀 Démo

[🔗 Voir la démo live](https://cac40-prediction.streamlit.app)

## 📊 Captures d'écran

### Dashboard Principal
![Dashboard](assets/dashboard.png)

### Prédiction
![Prediction](assets/prediction.png)

## 🛠️ Installation

### Prérequis
- Python 3.9 ou supérieur
- pip

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/yacineallam/cac40-prediction.git
cd cac40-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

L'application sera accessible à l'adresse http://localhost:8501

## 📁 Structure du Projet

```
cac40-prediction/
├── app.py                 # Application Streamlit principale
├── requirements.txt       # Dépendances Python
├── .streamlit/
│   └── config.toml       # Configuration Streamlit
├── models/               # Modèles entraînés (optionnel)
├── notebooks/            # Notebooks Jupyter
│   └── projet_cac40.ipynb
├── assets/               # Images et ressources
└── README.md
```

## 🔬 Méthodologie

### Données
- Source : Yahoo Finance (API yfinance)
- Indice : ^FCHI (CAC40)
- Période : 2015-2024 pour l'entraînement

### Feature Engineering
- **Rendements** : Variation journalière en %
- **Moyennes Mobiles** : MA5, MA10, MA20, MA50
- **RSI** : Relative Strength Index (14 périodes)
- **MACD** : Moving Average Convergence Divergence
- **Bandes de Bollinger** : Moyenne ± 2 écarts-types
- **Volatilité** : Écart-type glissant (20 jours)

### Modèle LSTM
- Architecture bidirectionnelle
- 100 unités LSTM
- Dropout 0.3 pour la régularisation
- Horizon de prédiction : 15 jours
- Seuil de mouvement significatif : 2.5%

### Performance
| Modèle | AUC-ROC |
|--------|---------|
| LSTM Bidirectionnel | 0.55 |
| Random Forest (baseline) | 0.54 |

## 📈 Indicateurs Techniques

### RSI (Relative Strength Index)
- < 30 : Survente (signal d'achat potentiel)
- > 70 : Surachat (signal de vente potentiel)

### MACD
- MACD > Signal : Momentum haussier
- MACD < Signal : Momentum baissier

### Moyennes Mobiles
- Prix > MA20 : Tendance haussière
- MA5 > MA20 : Potentiel "Golden Cross"

## 👨‍💻 Auteur

**Yacine ALLAM**
- Étudiant Ingénieur en Data Science
- ESIEA Paris (2022-2027)
- 📧 yacineallam00@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/yacine-allam)
- 💻 [GitHub](https://github.com/yacineallam)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## ⚠️ Avertissement

Cette application est fournie à des fins éducatives uniquement. Les prédictions ne constituent pas des conseils en investissement. Investir en bourse comporte des risques de perte en capital.

---

*Projet réalisé dans le cadre de ma formation en Data Science à l'ESIEA*

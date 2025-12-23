"""
CAC40 Trend Prediction - Streamlit Application
Author: Yacine ALLAM
Description: Application interactive pour la prédiction de tendance du CAC40
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="CAC40 Prediction | Yacine ALLAM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
    }
    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #422006 0%, #0f172a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data(start_date, end_date):
    """Charge les données du CAC40 depuis Yahoo Finance"""
    try:
        df = yf.download("^FCHI", start=start_date, end=end_date, progress=False)
        if len(df) > 0:
            # Aplatir les colonnes MultiIndex si nécessaire
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None


def calculate_indicators(df):
    """Calcule les indicateurs techniques"""
    data = df.copy()

    # Rendements
    data['Return'] = data['Close'].pct_change() * 100

    # Moyennes mobiles
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std * 2)

    # Volatilité
    data['Volatility'] = data['Return'].rolling(window=20).std()

    return data.dropna()


def create_price_chart(data, show_ma=True, show_bb=False):
    """Crée le graphique des prix avec indicateurs"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Prix CAC40', 'Volume', 'RSI')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='CAC40',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )

    # Moyennes mobiles
    if show_ma:
        colors = {'MA5': '#6366f1', 'MA10': '#a855f7', 'MA20': '#f59e0b', 'MA50': '#10b981'}
        for ma, color in colors.items():
            if ma in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[ma], name=ma, line=dict(color=color, width=1.5)),
                    row=1, col=1
                )

    # Bollinger Bands
    if show_bb:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper',
                      line=dict(color='rgba(99, 102, 241, 0.3)', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower',
                      line=dict(color='rgba(99, 102, 241, 0.3)', dash='dash'),
                      fill='tonexty', fillcolor='rgba(99, 102, 241, 0.1)'),
            row=1, col=1
        )

    # Volume
    colors = ['#10b981' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ef4444'
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, showlegend=False),
        row=2, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#a855f7', width=1.5)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
    )

    return fig


def create_macd_chart(data):
    """Crée le graphique MACD"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD'],
        name='MACD', line=dict(color='#6366f1', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD_Signal'],
        name='Signal', line=dict(color='#f59e0b', width=2)
    ))

    colors = ['#10b981' if val >= 0 else '#ef4444' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=data.index, y=data['MACD_Hist'],
        name='Histogram', marker_color=colors
    ))

    fig.update_layout(
        template='plotly_dark',
        title='MACD (Moving Average Convergence Divergence)',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_returns_distribution(data):
    """Crée l'histogramme des rendements"""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data['Return'],
        nbinsx=50,
        name='Rendements',
        marker_color='#6366f1',
        opacity=0.7
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="white")
    fig.add_vline(x=data['Return'].mean(), line_dash="solid", line_color="#10b981",
                  annotation_text=f"Moyenne: {data['Return'].mean():.2f}%")

    fig.update_layout(
        template='plotly_dark',
        title='Distribution des Rendements Journaliers',
        xaxis_title='Rendement (%)',
        yaxis_title='Fréquence',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
    )

    return fig


def simulate_prediction(data, horizon=15, threshold=2.5):
    """
    Simule une prédiction de tendance basée sur les indicateurs techniques.
    Note: Cette simulation utilise des règles simples pour la démonstration.
    Le modèle LSTM original nécessite TensorFlow pour fonctionner.
    """
    latest = data.iloc[-1]

    # Score basé sur les indicateurs
    score = 0
    signals = []

    # RSI
    if latest['RSI'] < 30:
        score += 2
        signals.append(("RSI", "Survente - Signal haussier", "bullish"))
    elif latest['RSI'] > 70:
        score -= 2
        signals.append(("RSI", "Surachat - Signal baissier", "bearish"))
    else:
        signals.append(("RSI", "Zone neutre", "neutral"))

    # MACD
    if latest['MACD'] > latest['MACD_Signal']:
        score += 1.5
        signals.append(("MACD", "MACD > Signal - Momentum haussier", "bullish"))
    else:
        score -= 1.5
        signals.append(("MACD", "MACD < Signal - Momentum baissier", "bearish"))

    # Tendance MA
    if latest['Close'] > latest['MA20']:
        score += 1
        signals.append(("MA20", "Prix > MA20 - Tendance haussière", "bullish"))
    else:
        score -= 1
        signals.append(("MA20", "Prix < MA20 - Tendance baissière", "bearish"))

    if latest['MA5'] > latest['MA20']:
        score += 1
        signals.append(("Croisement MA", "MA5 > MA20 - Golden cross potentiel", "bullish"))
    else:
        score -= 1
        signals.append(("Croisement MA", "MA5 < MA20 - Death cross potentiel", "bearish"))

    # Volatilité
    vol_percentile = (data['Volatility'] <= latest['Volatility']).mean() * 100
    if vol_percentile > 80:
        signals.append(("Volatilité", f"Élevée (percentile {vol_percentile:.0f}%)", "warning"))
    else:
        signals.append(("Volatilité", f"Normale (percentile {vol_percentile:.0f}%)", "neutral"))

    # Calcul de la probabilité
    probability = 0.5 + (score / 12)  # Normaliser entre 0 et 1
    probability = max(0.1, min(0.9, probability))  # Limiter entre 10% et 90%

    prediction = "HAUSSE" if probability > 0.5 else "BAISSE"
    confidence = abs(probability - 0.5) * 2  # Confiance entre 0 et 1

    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'signals': signals,
        'horizon': horizon,
        'threshold': threshold
    }


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 CAC40 Trend Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse technique et prédiction de tendance du CAC40</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 📈")
        st.markdown("### ⚙️ Configuration")

        # Sélection de la période
        period_options = {
            "1 mois": 30,
            "3 mois": 90,
            "6 mois": 180,
            "1 an": 365,
            "2 ans": 730,
            "5 ans": 1825
        }
        selected_period = st.selectbox("Période d'analyse", list(period_options.keys()), index=3)
        days = period_options[selected_period]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        st.markdown("---")
        st.markdown("### 📊 Indicateurs")
        show_ma = st.checkbox("Moyennes Mobiles", value=True)
        show_bb = st.checkbox("Bandes de Bollinger", value=False)

        st.markdown("---")
        st.markdown("### 🔮 Paramètres de Prédiction")
        horizon = st.slider("Horizon (jours)", 5, 30, 15)
        threshold = st.slider("Seuil mouvement (%)", 1.0, 5.0, 2.5, 0.5)

        st.markdown("---")
        st.markdown("### 👨‍💻 Auteur")
        st.markdown("""
        **Yacine ALLAM**
        Étudiant Ingénieur Data Science
        [LinkedIn](https://www.linkedin.com/in/yacineallam/) | [GitHub](https://github.com/YACINE-CODE16)
        """)

    # Chargement des données
    with st.spinner('Chargement des données CAC40...'):
        df = load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None or len(df) == 0:
        st.error("Impossible de charger les données. Veuillez réessayer.")
        return

    # Calcul des indicateurs
    data = calculate_indicators(df)

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = ((latest_price - prev_price) / prev_price) * 100

    with col1:
        st.metric(
            label="Prix Actuel",
            value=f"{latest_price:,.2f} €",
            delta=f"{change:+.2f}%"
        )

    with col2:
        st.metric(
            label="Volume",
            value=f"{data['Volume'].iloc[-1]:,.0f}",
            delta=f"{((data['Volume'].iloc[-1] / data['Volume'].mean()) - 1) * 100:+.1f}% vs moy"
        )

    with col3:
        st.metric(
            label="RSI (14)",
            value=f"{data['RSI'].iloc[-1]:.1f}",
            delta="Surachat" if data['RSI'].iloc[-1] > 70 else ("Survente" if data['RSI'].iloc[-1] < 30 else "Neutre")
        )

    with col4:
        volatility = data['Volatility'].iloc[-1]
        st.metric(
            label="Volatilité",
            value=f"{volatility:.2f}%",
            delta="Élevée" if volatility > data['Volatility'].quantile(0.8) else "Normale"
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Graphique", "🔮 Prédiction", "📊 Analyse", "ℹ️ À propos"])

    with tab1:
        st.plotly_chart(create_price_chart(data, show_ma, show_bb), use_container_width=True)
        st.plotly_chart(create_macd_chart(data), use_container_width=True)

    with tab2:
        st.markdown("### 🔮 Prédiction de Tendance")

        # Simulation de prédiction
        pred = simulate_prediction(data, horizon, threshold)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Résultat de la prédiction
            color = "#10b981" if pred['prediction'] == "HAUSSE" else "#ef4444"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}22 0%, #0f172a 100%);
                        border: 2px solid {color}; border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 0.5rem;">
                    Prédiction à {pred['horizon']} jours
                </div>
                <div style="font-size: 3rem; font-weight: 700; color: {color};">
                    {pred['prediction']}
                </div>
                <div style="font-size: 1.5rem; color: white; margin-top: 0.5rem;">
                    {pred['probability']*100:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 0.5rem;">
                    Confiance: {pred['confidence']*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Signaux Techniques")
            for signal in pred['signals']:
                icon = "🟢" if signal[2] == "bullish" else ("🔴" if signal[2] == "bearish" else ("⚠️" if signal[2] == "warning" else "⚪"))
                st.markdown(f"**{icon} {signal[0]}**: {signal[1]}")

        st.markdown("---")

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Avertissement</strong><br>
            Cette prédiction est basée sur une simulation utilisant des indicateurs techniques.
            Elle ne constitue pas un conseil en investissement. Le modèle LSTM original
            (AUC: 0.55) nécessite TensorFlow pour fonctionner.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 📊 Analyse Statistique")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_returns_distribution(data), use_container_width=True)

        with col2:
            # Statistiques descriptives
            stats = data['Return'].describe()
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Statistique', 'Valeur'],
                    fill_color='#1e293b',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Moyenne', 'Écart-type', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis'],
                        [f"{stats['mean']:.3f}%", f"{stats['std']:.3f}%", f"{stats['min']:.3f}%",
                         f"{stats['25%']:.3f}%", f"{stats['50%']:.3f}%", f"{stats['75%']:.3f}%",
                         f"{stats['max']:.3f}%", f"{data['Return'].skew():.3f}", f"{data['Return'].kurtosis():.3f}"]
                    ],
                    fill_color='#0f172a',
                    font=dict(color='#e2e8f0', size=13),
                    align='left',
                    height=30
                )
            )])
            fig.update_layout(
                title='Statistiques des Rendements',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Corrélation avec le volume
        st.markdown("#### Corrélation Prix-Volume")
        fig = px.scatter(
            data, x='Volume', y='Return',
            color='Return', color_continuous_scale='RdYlGn',
            title='Relation Volume / Rendement'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.8)',
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### ℹ️ À propos du projet")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            #### Prédiction de Tendance CAC40 avec LSTM

            Ce projet utilise un réseau de neurones LSTM bidirectionnel pour prédire
            la tendance du CAC40 sur un horizon de 15 jours.

            **Caractéristiques du modèle:**
            - Architecture LSTM bidirectionnelle (100 unités)
            - Features: Indicateurs techniques (RSI, MACD, MA, Bollinger)
            - Horizon de prédiction: 15 jours
            - Seuil de mouvement significatif: 2.5%

            **Performance:**
            - AUC LSTM: 0.55
            - AUC Random Forest (baseline): 0.54

            **Technologies utilisées:**
            - Python, TensorFlow/Keras
            - Streamlit, Plotly
            - yfinance, Pandas, NumPy
            """)

        with col2:
            st.markdown("""
            #### Contact

            **Yacine ALLAM**
            Étudiant Ingénieur
            ESIEA Paris

            📧 yacineallam00@gmail.com
            🔗 [LinkedIn](https://www.linkedin.com/in/yacineallam/)
            💻 [GitHub](https://github.com/YACINE-CODE16)

            ---

            *Projet réalisé dans le cadre de ma formation en Data Science*
            """)

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <strong>📚 Sources de données</strong><br>
            Les données historiques du CAC40 proviennent de Yahoo Finance via l'API yfinance.
            Elles sont mises à jour en temps réel pendant les heures de marché.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

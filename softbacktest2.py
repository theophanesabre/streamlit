# -*- coding: utf-8 -*-
# app.py (v2.8 - Ajout Strat EMA Cross + ADX, Suppression Spread, UI/Calculs/Plot Conditionnels)

import streamlit as st
import pandas as pd
import ta # Utilise pandas_ta implicitement si installé, sinon ta-lib si disponible
import matplotlib.pyplot as plt
import matplotlib
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import traceback

# Configuration Matplotlib et Page Streamlit
matplotlib.use('Agg')
st.set_page_config(layout="wide", page_title="Lovecrash Backtester v2.8") # Version mise à jour

# --- Injection CSS et Titre ---
# (Code CSS identique - omis pour la lisibilité)
font_path = "PASTOROF.ttf"; font_css = "" # Chemin vers votre police personnalisée si utilisée
try:
    # Essayer de générer le CSS pour la police personnalisée et le thème sombre
    if os.path.exists(font_path):
        # Note: L'intégration directe de fichiers locaux peut être limitée dans Streamlit Cloud.
        # Pour une police web, utiliser @import url(...) dans le CSS est plus robuste.
        font_css = f"""
        <style>
            @font-face {{
                font-family: 'PASTOROF';
                src: url('{font_path}') format('truetype'); /* Peut nécessiter ajustement selon déploiement */
            }}
            h1[data-testid="stHeading"], .stApp > header h1 {{
                font-family: 'PASTOROF', sans-serif !important; /* Fallback */
                font-size: 3rem !important;
            }}
            body {{
                font-family: 'Helvetica Neue', sans-serif; /* Police fallback plus standard */
            }}
            div.main {{
                background-color: #000000 !important; /* Fond noir */
                color: #FFFFFF !important; /* Texte blanc */
            }}
            /* Ciblage plus spécifique pour les éléments Streamlit */
            div[data-testid="stAppViewContainer"], .stApp {{
                color: #FFFFFF !important;
            }}
            div[data-testid="stNumberInput"] label,
            div[data-testid="stTextInput"] label,
            div[data-testid="stRadio"] label,
            div[data-testid="stCheckbox"] label,
            div[data-testid="stToggle"] label,
            div[data-testid="stSelectbox"] label {{ /* Ajout Selectbox */
                color: #EEEEEE !important; /* Labels des widgets */
            }}
            div[data-testid="stMetric"] {{
                background-color: #111111 !important; /* Fond métriques */
                border-radius: 5px;
                padding: 10px;
                color: #FFFFFF !important;
            }}
            div[data-testid="stMetric"] > label {{ /* Label des métriques */
                color: #AAAAAA !important;
            }}
            div[data-testid="stMetric"] > div:nth-of-type(2) {{ /* Valeur des métriques */
                color: #FFFFFF !important;
            }}
            /* Améliorer lisibilité DataFrames */
             .stDataFrame {{ color: #333; }} /* Texte sombre pour contenu tableau */

        </style>
        """
    else:
         # CSS Thème Sombre Minimal si police non trouvée
         font_css = """
         <style>
             h1[data-testid="stHeading"], .stApp > header h1 { font-size: 3rem !important; }
             body { font-family: 'Helvetica Neue', sans-serif; }
             div.main { background-color: #000000 !important; color: #FFFFFF !important; }
             div[data-testid="stAppViewContainer"], .stApp { color: #FFFFFF !important; }
             div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label,
             div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label,
             div[data-testid="stToggle"] label, div[data-testid="stSelectbox"] label { color: #EEEEEE !important; }
             div[data-testid="stMetric"] { background-color: #111111 !important; border-radius: 5px; padding: 10px; color: #FFFFFF !important; }
             div[data-testid="stMetric"] > label { color: #AAAAAA !important; }
             div[data-testid="stMetric"] > div:nth-of-type(2) { color: #FFFFFF !important; }
             .stDataFrame { color: #333; }
         </style>
         """
    st.markdown(font_css, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Police perso/CSS non chargé: {e}", icon="🎨")
st.title("Lovecrash")
# --- Fin Style ---

# ==============================================================
# Fonction de Calcul des Statistiques
# ==============================================================
# (Identique à la version précédente v2.7)
def calculate_statistics(trade_history, equity_curve, initial_equity):
    """Calcule les statistiques clés du backtest."""
    stats = {}
    number_of_trades = len(trade_history)
    stats['Number of Trades'] = number_of_trades
    stats['First Trade Date'] = None
    stats['Last Trade Date'] = None
    stats['Total Profit'] = 0
    stats['Final Equity'] = initial_equity
    stats['Profit (%)'] = 0
    stats['Winning Trades (%)'] = 0
    stats['Max Drawdown (%)'] = 0
    stats['Max Consecutive Losing Trades'] = 0
    stats['Average Consecutive Losing Trades'] = 0
    stats['Average Profit per Trade'] = 0
    stats['Profit Factor'] = 0

    if number_of_trades == 0: # Vérifier seulement le nombre de trades initialement
         # Si pas de trades, retourner stats à zéro mais garder équité initiale
         stats['Final Equity'] = initial_equity
         return stats

    # Assurer que trade_history est un DataFrame pour les calculs suivants
    if not isinstance(trade_history, pd.DataFrame):
        trade_history_df = pd.DataFrame(trade_history)
    else:
        trade_history_df = trade_history

    if trade_history_df.empty:
        stats['Final Equity'] = initial_equity
        return stats

    try:
        # Convertir les temps si ce ne sont pas déjà des Timestamps
        if not pd.api.types.is_datetime64_any_dtype(trade_history_df['entry_time']):
             trade_history_df['entry_time'] = pd.to_datetime(trade_history_df['entry_time'])
        if not pd.api.types.is_datetime64_any_dtype(trade_history_df['exit_time']):
             trade_history_df['exit_time'] = pd.to_datetime(trade_history_df['exit_time'])

        trade_history_df_sorted = trade_history_df.sort_values(by='exit_time')

        stats['First Trade Date'] = trade_history_df_sorted['entry_time'].iloc[0]
        stats['Last Trade Date'] = trade_history_df_sorted['exit_time'].iloc[-1]

        total_profit = trade_history_df_sorted['profit'].sum()
        stats['Total Profit'] = total_profit
        final_equity = initial_equity + total_profit
        stats['Final Equity'] = final_equity
        stats['Profit (%)'] = (total_profit / initial_equity) * 100 if initial_equity > 0 else 0

        winning_trades = trade_history_df_sorted[trade_history_df_sorted['profit'] > 0]
        losing_trades = trade_history_df_sorted[trade_history_df_sorted['profit'] <= 0]
        stats['Winning Trades (%)'] = len(winning_trades) / number_of_trades * 100 if number_of_trades > 0 else 0

        # Calcul Drawdown seulement si equity_curve n'est pas vide et a un index Datetime
        if not equity_curve.empty and isinstance(equity_curve.index, pd.DatetimeIndex):
            temp_equity_curve_index = equity_curve.index
            start_time = temp_equity_curve_index.min() - pd.Timedelta(seconds=1)
            temp_equity_curve = pd.concat([pd.Series([initial_equity], index=[start_time]), equity_curve])
            temp_equity_curve = temp_equity_curve.sort_index()

            max_drawdown, current_peak = 0, initial_equity
            for val in temp_equity_curve:
                current_peak = max(current_peak, val)
                drawdown = (current_peak - val) / current_peak * 100 if current_peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            stats['Max Drawdown (%)'] = max_drawdown
        else:
             stats['Max Drawdown (%)'] = 0 # Pas de calcul si courbe d'équité invalide/vide

        # Calcul Streaks Perdantes
        losing_streak, max_losing_streak, losing_streak_lengths = 0, 0, []
        for profit in trade_history_df_sorted['profit']:
            if profit <= 0: losing_streak += 1
            else:
                if losing_streak > 0: losing_streak_lengths.append(losing_streak)
                max_losing_streak = max(max_losing_streak, losing_streak)
                losing_streak = 0
        if losing_streak > 0: # Compter la dernière streak
            losing_streak_lengths.append(losing_streak)
            max_losing_streak = max(max_losing_streak, losing_streak)

        stats['Max Consecutive Losing Trades'] = max_losing_streak
        stats['Average Consecutive Losing Trades'] = sum(losing_streak_lengths) / len(losing_streak_lengths) if losing_streak_lengths else 0

        stats['Average Profit per Trade'] = total_profit / number_of_trades if number_of_trades > 0 else 0

        gross_profit = winning_trades['profit'].sum()
        gross_loss = abs(losing_trades['profit'].sum())
        stats['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    except Exception as e:
        st.error(f"Erreur calcul statistiques: {e}")
        st.error(traceback.format_exc())
        # Retourner les stats par défaut ou partielles
        default_stats = {k: 0 for k in ['Number of Trades', 'Total Profit', 'Profit (%)', 'Winning Trades (%)', 'Max Drawdown (%)', 'Max Consecutive Losing Trades', 'Average Consecutive Losing Trades', 'Average Profit per Trade', 'Profit Factor']}
        default_stats['Final Equity'] = initial_equity
        return default_stats

    return stats

# ==============================================================
# Fonction Chargement Données & Indicateurs (MAJ pour Cross EMA)
# ==============================================================
@st.cache_data
def load_data_and_indicators(url,
                             calc_ema=False, short_ema_p=50, long_ema_p=200,
                             calc_rsi=False, rsi_p=14,
                             calc_atr=False, atr_p=14,
                             calc_divergence=False, div_lookback_p=30,
                             calc_adx=False, adx_p=14,
                             calc_reversal_ma=False, reversal_ma_p=100
                             ):
    """
    Charge les données et calcule UNIQUEMENT les indicateurs nécessaires.
    Calcule les croisements EMA si les EMAs sont calculées.
    """
    active_calcs = []
    if calc_ema: active_calcs.append(f"EMA({short_ema_p},{long_ema_p})+Cross") # Ajout +Cross
    if calc_reversal_ma: active_calcs.append(f"SMA({reversal_ma_p})")
    if calc_rsi: active_calcs.append(f"RSI({rsi_p})")
    if calc_atr: active_calcs.append(f"ATR({atr_p})")
    if calc_divergence: active_calcs.append(f"Div({div_lookback_p})")
    if calc_adx: active_calcs.append(f"ADX({adx_p})")
    st.write(f"CACHE MISS: Chargement/Prep Données ({', '.join(active_calcs) if active_calcs else 'Aucun Indicateur Requis'})...")

    calculated_indic_cols = []
    try:
        df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
        df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()

        numeric_cols=['Open','High','Low','Close']
        df.dropna(subset=numeric_cols, inplace=True)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)

        min_rows_needed = 1 # Minimum de base
        if calc_ema: min_rows_needed = max(min_rows_needed, long_ema_p + 1) # +1 pour shift cross
        if calc_reversal_ma: min_rows_needed = max(min_rows_needed, reversal_ma_p)
        if calc_rsi: min_rows_needed = max(min_rows_needed, rsi_p)
        if calc_atr: min_rows_needed = max(min_rows_needed, atr_p)
        if calc_adx: min_rows_needed = max(min_rows_needed, adx_p * 2) # ADX a besoin de plus de données
        if calc_divergence: min_rows_needed = max(min_rows_needed, div_lookback_p + rsi_p) # Approx

        if len(df) < min_rows_needed:
             st.error(f"Données insuffisantes ({len(df)} lignes) pour calculer indicateurs (min requis: ~{min_rows_needed}).")
             return pd.DataFrame()

        # --- Calculs Indicateurs Strictement Conditionnels ---
        if calc_ema:
            df['ema_short']=ta.trend.ema_indicator(df['Close'],window=short_ema_p)
            df['ema_long']=ta.trend.ema_indicator(df['Close'],window=long_ema_p)
            # --- Ajout Calcul Croisements ---
            df['ema_cross_up'] = (df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1))
            df['ema_cross_down'] = (df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1))
            calculated_indic_cols.extend(['ema_short', 'ema_long', 'ema_cross_up', 'ema_cross_down'])
        else:
            df['ema_cross_up'] = False; df['ema_cross_down'] = False # Assurer existence

        if calc_reversal_ma and reversal_ma_p > 0:
            df['reversal_ma'] = ta.trend.sma_indicator(df['Close'], window=reversal_ma_p)
            calculated_indic_cols.append('reversal_ma')
        else: df['reversal_ma'] = pd.NA

        if calc_rsi:
            df['rsi']=ta.momentum.rsi(df['Close'],window=rsi_p)
            calculated_indic_cols.append('rsi')

        if calc_atr:
            df['atr']=ta.volatility.average_true_range(df['High'],df['Low'],df['Close'],window=atr_p)
            calculated_indic_cols.append('atr')
        else: df['atr'] = pd.NA

        if calc_adx:
            if df[['High', 'Low', 'Close']].isnull().any().any():
                 st.warning("NaNs détectés dans High/Low/Close avant calcul ADX, possible erreur.")
                 df['adx']=pd.NA; df['di_pos']=pd.NA; df['di_neg']=pd.NA
            else:
                 # Utilisation de pandas_ta.adx si disponible (plus robuste) ou ta si besoin
                 try:
                      # Essayer avec pandas_ta si installé
                      import pandas_ta as pta
                      adx_df = df.ta.adx(length=adx_p) # pandas_ta retourne un DataFrame
                      df['adx'] = adx_df[f'ADX_{adx_p}']
                      df['di_pos'] = adx_df[f'DMP_{adx_p}']
                      df['di_neg'] = adx_df[f'DMN_{adx_p}']
                 except ImportError:
                      # Fallback sur 'ta' si pandas_ta n'est pas là
                      st.info("Utilisation de la librairie 'ta' pour l'ADX (pandas_ta recommandé).")
                      df['adx']=ta.trend.adx(df['High'],df['Low'],df['Close'],window=adx_p)
                      df['di_pos']=ta.trend.adx_pos(df['High'],df['Low'],df['Close'],window=adx_p)
                      df['di_neg']=ta.trend.adx_neg(df['High'],df['Low'],df['Close'],window=adx_p)

                 calculated_indic_cols.extend(['adx', 'di_pos', 'di_neg'])
        else:
            df['adx']=pd.NA; df['di_pos']=pd.NA; df['di_neg']=pd.NA

        if calc_divergence:
            if 'rsi' in calculated_indic_cols and not df['rsi'].isnull().all():
                if div_lookback_p > 0:
                    price_diff=df['Close'].diff(div_lookback_p)
                    rsi_diff=df['rsi'].diff(div_lookback_p)
                    df['bullish_divergence']=(price_diff < 0) & (rsi_diff > 0)
                    df['bearish_divergence']=(price_diff > 0) & (rsi_diff < 0)
                    calculated_indic_cols.extend(['bullish_divergence', 'bearish_divergence'])
                else: df['bullish_divergence']=False; df['bearish_divergence']=False
            else:
                 st.warning("Calcul divergence impossible car RSI non disponible ou invalide.")
                 df['bullish_divergence']=False; df['bearish_divergence']=False
        else:
            df['bullish_divergence']=False; df['bearish_divergence']=False

        # Nettoyage final
        cols_to_dropna = [col for col in calculated_indic_cols if col in df.columns and df[col].isnull().any()]
        if cols_to_dropna:
             # Attention: dropna peut supprimer beaucoup de lignes au début
             initial_len = len(df)
             df.dropna(subset=cols_to_dropna, inplace=True)
             # st.write(f"Dropna sur {cols_to_dropna}, lignes supprimées: {initial_len - len(df)}") # Debug optionnel

        st.write("CACHE MISS: Fin chargement et préparation.")
        return df

    except Exception as e:
        st.error(f"Erreur irrécupérable chargement/préparation données: {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame()


# ==============================================================
# Fonction Principale de Backtesting (MAJ pour EMA Cross ADX)
# ==============================================================
def backtest_strategy(df_processed, initial_equity=5000,
                      # --- Choix Stratégie ---
                      strategy_type='trend_divergence', # 'trend_divergence', 'reversal_ma_div', 'ema_cross_adx'
                      # --- Paramètres communs ---
                      risk_percentage=0.005, # Utilisé si sizing_type='risk_pct'
                      sizing_type='risk_pct',  # Forcé à 'risk_pct' pour ema_cross_adx
                      fixed_lot_size=0.1,   # Ignoré si sizing_type='risk_pct'
                      sl_type='percentage', # Forcé à 'percentage' pour ema_cross_adx
                      stop_loss_percentage=0.005, # Utilisé si sl_type='percentage' (0.5% = 0.005)
                      atr_multiplier_sl=2.0, # Ignoré si sl_type='percentage'
                      take_profit_multiplier=4.0, # RR Ratio
                      atr_threshold=0.0, one_trade_at_a_time=True,
                      # --- Paramètres Indicateurs ---
                      ema_short_period=50, ema_long_period=200, rsi_length=14,
                      rsi_oversold=30, rsi_overbought=70, atr_period=14, adx_period=14,
                      reversal_ma_period=100, div_lookback_period=30,
                      adx_threshold=25.0, # Ajout pour EMA Cross ADX strat
                      # --- Options spécifiques Tendance/Div ---
                      use_ema_rsi_signal=True, use_bullish_div=False, use_bearish_div=False,
                      use_points_system=False,
                      points_ema_rsi_long=1, points_ema_rsi_short=1, points_bull_div=1,
                      points_bear_div=1, long_score_threshold=1, short_score_threshold=1,
                      signal_validity_bars=1,
                      # --- Filtres ---
                      use_adx_filter=False, # Filtre ADX général (différent du seuil dans EMA Cross)
                      # adx_threshold utilisé pour le filtre général est passé séparément
                      use_min_profit_points_filter=False, min_profit_points_threshold=0.0,
                      progress_placeholder=None ):
    """Effectue le backtest selon la stratégie choisie."""

    if df_processed.empty:
        st.error("Impossible lancer backtest: DataFrame vide.")
        return pd.DataFrame(), pd.Series(dtype=float), {}, None

    # Initialisations
    fig = None; closed_trades_history = []; equity_history = [initial_equity]
    equity = initial_equity; open_positions = []; trade_id_counter = 0
    total_rows = len(df_processed)
    st.write(f"Début boucle backtesting ({total_rows} bougies) - Stratégie: {strategy_type}")

    # Initialisation états validité (seulement pour 'trend_divergence')
    ema_rsi_long_last_active_idx = -signal_validity_bars; ema_rsi_short_last_active_idx = -signal_validity_bars
    bull_div_last_active_idx = -signal_validity_bars; bear_div_last_active_idx = -signal_validity_bars
    prev_long_signal_ema_rsi = False; prev_short_signal_ema_rsi = False
    prev_long_signal_div = False; prev_short_signal_div = False

    # --- Vérification colonnes nécessaires pour la stratégie active ---
    required_cols = ['Open', 'High', 'Low', 'Close']
    if strategy_type == 'trend_divergence':
        if use_ema_rsi_signal: required_cols.extend(['ema_short', 'ema_long', 'rsi'])
        if use_bullish_div or use_bearish_div: required_cols.extend(['rsi', 'bullish_divergence', 'bearish_divergence'])
    elif strategy_type == 'reversal_ma_div':
        required_cols.extend(['reversal_ma', 'rsi', 'bullish_divergence', 'bearish_divergence'])
    elif strategy_type == 'ema_cross_adx':
        required_cols.extend(['ema_short', 'ema_long', 'ema_cross_up', 'ema_cross_down', 'adx'])

    # Ajouter colonnes pour filtres/SL/TP actifs
    if sl_type=='atr' or atr_threshold > 0: required_cols.append('atr')
    if use_adx_filter: required_cols.extend(['adx']) # ADX déjà ajouté si ema_cross_adx

    missing_cols = [col for col in set(required_cols) if col not in df_processed.columns or df_processed[col].isnull().all()]
    if missing_cols:
        st.error(f"Erreur: Colonnes critiques manquantes/vides pour la config: {', '.join(missing_cols)}.")
        return pd.DataFrame(), pd.Series(dtype=float), {}, None

    # --- Boucle Principale sur les bougies ---
    for i, (index, row) in enumerate(df_processed.iterrows()):
        # Affichage Progression
        if progress_placeholder and (i % 500 == 0 or i == total_rows - 1):
             prog = float(i+1)/total_rows; perc = min(int(prog*100),100)
             progress_placeholder.text(f"Progression: {perc}%")

        # Extraction données bougie courante
        signal_price=row['Close']; current_high=row['High']; current_low=row['Low']
        current_atr=row.get('atr', 0); # Utilise 0 si NaN ou non calculé
        adx_val=row.get('adx',0);     # Utilise 0 si NaN ou non calculé

        if pd.isna(signal_price) or signal_price <= 0: continue

        # --- Logique de Sortie des positions ouvertes ---
        # (Identique à la version précédente)
        positions_to_remove = []
        for position in open_positions:
             exit_price = None
             pos_id=position['id']; pos_entry_price=position['entry_price']; pos_type=position['type']
             pos_sl=position['stop_loss']; pos_tp=position['take_profit']; pos_size=position['size']
             pos_entry_time=position['entry_time']

             if pos_type=='long':
                 if current_low <= pos_sl: exit_price = pos_sl
                 elif current_high >= pos_tp: exit_price = pos_tp
             elif pos_type=='short':
                 if current_high >= pos_sl: exit_price = pos_sl
                 elif current_low <= pos_tp: exit_price = pos_tp

             if exit_price is not None:
                 if pos_type=='long': profit = (exit_price - pos_entry_price) * pos_size
                 else: profit = (pos_entry_price - exit_price) * pos_size
                 equity += profit
                 equity = max(equity, 0)
                 closed_trades_history.append({
                     'trade_id':pos_id, 'entry_time':pos_entry_time, 'entry_price':pos_entry_price,
                     'entry_type':pos_type, 'size':pos_size, 'stop_loss':pos_sl, 'take_profit':pos_tp,
                     'exit_time':index, 'exit_price':exit_price, 'profit':profit
                 })
                 equity_history.append(equity)
                 positions_to_remove.append(position)
        for closed_pos in positions_to_remove: open_positions.remove(closed_pos)


        # --- Logique d'Entrée ---
        # Filtres communs
        is_atr_valid_for_use = not pd.isna(current_atr) and current_atr > 1e-9
        if sl_type=='atr' and not is_atr_valid_for_use: continue
        if atr_threshold > 0 and (not is_atr_valid_for_use or current_atr < atr_threshold): continue
        # Le filtre ADX général est différent du seuil utilisé dans EMA Cross ADX
        if use_adx_filter and (pd.isna(adx_val) or adx_val < adx_threshold): continue

        # --- Calcul Signaux basé sur la Stratégie Choisie ---
        final_long_signal = False
        final_short_signal = False

        # ========================================
        # STRATÉGIE 1: TENDANCE / DIVERGENCE
        # ========================================
        if strategy_type == 'trend_divergence':
            # --- Calcul signaux individuels ---
            current_long_signal_ema_rsi = False; current_short_signal_ema_rsi = False
            if use_ema_rsi_signal:
                ema_short=row.get('ema_short'); ema_long=row.get('ema_long'); rsi=row.get('rsi')
                if not pd.isna(ema_short) and not pd.isna(ema_long) and not pd.isna(rsi):
                    current_long_signal_ema_rsi = signal_price > ema_short and ema_short > ema_long and rsi < rsi_oversold
                    current_short_signal_ema_rsi = signal_price < ema_short and ema_short < ema_long and rsi > rsi_overbought

            current_long_signal_div = row.get('bullish_divergence', False) if use_bullish_div else False
            current_short_signal_div = row.get('bearish_divergence', False) if use_bearish_div else False

            # --- Mise à jour index dernière activation ---
            if current_long_signal_ema_rsi and not prev_long_signal_ema_rsi: ema_rsi_long_last_active_idx = i
            if current_short_signal_ema_rsi and not prev_short_signal_ema_rsi: ema_rsi_short_last_active_idx = i
            if current_long_signal_div and not prev_long_signal_div: bull_div_last_active_idx = i
            if current_short_signal_div and not prev_short_signal_div: bear_div_last_active_idx = i

            # --- Combinaison signaux ---
            if use_points_system:
                long_score = 0; short_score = 0
                if use_ema_rsi_signal and (i - ema_rsi_long_last_active_idx < signal_validity_bars): long_score += points_ema_rsi_long
                if use_bullish_div and (i - bull_div_last_active_idx < signal_validity_bars): long_score += points_bull_div
                if use_ema_rsi_signal and (i - ema_rsi_short_last_active_idx < signal_validity_bars): short_score += points_ema_rsi_short
                if use_bearish_div and (i - bear_div_last_active_idx < signal_validity_bars): short_score += points_bear_div
                final_long_signal = (long_score >= long_score_threshold)
                final_short_signal = (short_score >= short_score_threshold)
            else: # Logique OU simple
                signal_active_long_ema = use_ema_rsi_signal and (i - ema_rsi_long_last_active_idx < signal_validity_bars)
                signal_active_long_div = use_bullish_div and (i - bull_div_last_active_idx < signal_validity_bars)
                final_long_signal = signal_active_long_ema or signal_active_long_div
                signal_active_short_ema = use_ema_rsi_signal and (i - ema_rsi_short_last_active_idx < signal_validity_bars)
                signal_active_short_div = use_bearish_div and (i - bear_div_last_active_idx < signal_validity_bars)
                final_short_signal = signal_active_short_ema or signal_active_short_div

            # --- Mémoriser état ---
            prev_long_signal_ema_rsi = current_long_signal_ema_rsi; prev_short_signal_ema_rsi = current_short_signal_ema_rsi
            prev_long_signal_div = current_long_signal_div; prev_short_signal_div = current_short_signal_div

        # ========================================
        # STRATÉGIE 2: REVERSAL (MA + RSI Divergence)
        # ========================================
        elif strategy_type == 'reversal_ma_div':
            # --- Logique Reversal ---
            current_price = signal_price
            ma_value = row.get('reversal_ma')
            bull_div = row.get('bullish_divergence', False)
            bear_div = row.get('bearish_divergence', False)
            if not pd.isna(current_price) and not pd.isna(ma_value):
                if current_price < ma_value and bull_div: final_long_signal = True
                elif current_price > ma_value and bear_div: final_short_signal = True

        ### MODIFICATION START ###
        # ========================================
        # STRATÉGIE 3: CROISEMENT EMA + ADX
        # ========================================
        elif strategy_type == 'ema_cross_adx':
             # --- Logique EMA Cross + ADX ---
             is_cross_up = row.get('ema_cross_up', False)
             is_cross_down = row.get('ema_cross_down', False)
             is_adx_strong = not pd.isna(adx_val) and adx_val > adx_threshold # Utilise le seuil spécifique

             if is_cross_up and is_adx_strong:
                  final_long_signal = True
             elif is_cross_down and is_adx_strong:
                  final_short_signal = True
        ### MODIFICATION END ###

        # --- FIN Calcul Signaux ---

        # --- Prise de Position ---
        can_enter = True if not one_trade_at_a_time else (len(open_positions) == 0)

        if can_enter and (final_long_signal or final_short_signal):
            is_long = final_long_signal
            is_short = final_short_signal and not is_long

            if is_long or is_short:
                if equity <= 0: st.warning("Équité <= 0. Arrêt du backtest."); break

                actual_entry_price, stop_loss_price, risk_per_unit = None, None, 0.0

                # Prix d'Entrée (sans spread)
                actual_entry_price = signal_price
                if actual_entry_price is None or pd.isna(actual_entry_price) or actual_entry_price <= 0: continue

                # --- Calcul Stop Loss (Adapté pour forcer % si ema_cross_adx) ---
                current_sl_type = sl_type
                if strategy_type == 'ema_cross_adx':
                     current_sl_type = 'percentage' # Forcer SL % pour cette stratégie

                try:
                    if current_sl_type == 'percentage':
                        sl_pct_effective = stop_loss_percentage # = sl_pct / 100 déjà fait dans l'UI
                        if pd.isna(sl_pct_effective) or sl_pct_effective <= 0: continue # Vérif % SL valide
                        stop_loss_price = actual_entry_price * (1 - sl_pct_effective if is_long else 1 + sl_pct_effective)
                    elif current_sl_type == 'atr':
                        if not is_atr_valid_for_use: continue # ATR doit être valide
                        sl_offset = atr_multiplier_sl * current_atr
                        stop_loss_price = actual_entry_price - sl_offset if is_long else actual_entry_price + sl_offset
                    else: continue
                except Exception as e: continue

                # Vérification SL valide
                if stop_loss_price is None or pd.isna(stop_loss_price): continue
                if is_long and stop_loss_price >= actual_entry_price: continue
                if is_short and stop_loss_price <= actual_entry_price: continue

                # Calcul Risque
                risk_per_unit = abs(actual_entry_price - stop_loss_price)
                if risk_per_unit <= 1e-9: continue

                # Calcul Take Profit
                try:
                    # take_profit_multiplier est le RR Ratio
                    tp_offset = risk_per_unit * take_profit_multiplier
                    take_profit_price = actual_entry_price + tp_offset if is_long else actual_entry_price - tp_offset
                    if take_profit_price is None or pd.isna(take_profit_price): continue
                    if is_long and take_profit_price <= actual_entry_price: continue
                    if is_short and take_profit_price >= actual_entry_price: continue
                except Exception as e: continue

                # Filtre Min Profit Points
                execute_this_trade = True
                if use_min_profit_points_filter:
                    potential_profit_points = abs(take_profit_price - actual_entry_price)
                    if potential_profit_points < min_profit_points_threshold: execute_this_trade = False

                # --- Exécution si tous filtres passés ---
                if execute_this_trade:
                    # --- Calcul Sizing (Adapté pour forcer % si ema_cross_adx) ---
                    current_sizing_type = sizing_type
                    if strategy_type == 'ema_cross_adx':
                        current_sizing_type = 'risk_pct' # Forcer Sizing %

                    position_size = 0.0
                    try:
                        if current_sizing_type == 'risk_pct':
                            if equity > 0: position_size = (equity * risk_percentage) / risk_per_unit
                            else: position_size = 0
                        elif current_sizing_type == 'fixed_lot':
                            position_size = fixed_lot_size
                        else: continue
                    except Exception as e: continue

                    if position_size > 1e-9:
                        trade_id_counter += 1
                        new_position = {
                            'id': trade_id_counter, 'entry_time': index,
                            'entry_price': actual_entry_price, 'type': 'long' if is_long else 'short',
                            'size': position_size, 'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price
                        }
                        open_positions.append(new_position)

    # --- Finalisation Après la Boucle ---
    # (Identique)
    st.write("Fin boucle backtesting.")
    trade_history_df=pd.DataFrame(closed_trades_history)
    equity_curve_s = pd.Series(dtype=float) # Initialiser vide
    if not trade_history_df.empty:
         st.write("Calcul stats & graphiques...")
         try:
             if not equity_history: equity_curve_s = pd.Series(dtype=float)
             else:
                 # Créer l'index à partir de l'historique des trades
                 equity_curve_index = pd.to_datetime(trade_history_df['exit_time'])
                 # S'assurer qu'il y a autant de points d'équité que de trades + 1 (capital initial)
                 if len(equity_history[1:]) == len(equity_curve_index):
                      equity_curve_s = pd.Series(equity_history[1:], index=equity_curve_index)
                      equity_curve_s = equity_curve_s[~equity_curve_s.index.duplicated(keep='last')].sort_index()
                 else:
                      st.warning(f"Incohérence taille historique équité ({len(equity_history)}) vs trades ({len(trade_history_df)}). Graphique équité peut être incorrect.")
                      # Essayer de construire quand même si possible
                      min_len = min(len(equity_history[1:]), len(equity_curve_index))
                      equity_curve_s = pd.Series(equity_history[1:min_len+1], index=equity_curve_index[:min_len])
                      if not equity_curve_s.empty:
                          equity_curve_s = equity_curve_s[~equity_curve_s.index.duplicated(keep='last')].sort_index()
                      else: equity_curve_s = pd.Series(dtype=float)

             stats = calculate_statistics(trade_history_df, equity_curve_s, initial_equity)
             # Générer graphique seulement si la courbe d'équité n'est pas vide
             if not equity_curve_s.empty:
                  fig, ax = plt.subplots(figsize=(12, 6)); ax.plot(equity_curve_s.index, equity_curve_s.values, label='Equity Curve', marker='.', linestyle='-')
                  ax.set_title('Backtest Equity Progression'); ax.set_xlabel('Time'); ax.set_ylabel('Equity ($)')
                  ax.grid(True); ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
             else: fig = None # Pas de graphique si courbe vide
             st.write("Calculs terminés.")
         except Exception as final_calc_err:
             st.error(f"Erreur lors finalisation (stats/graph équité): {final_calc_err}")
             st.error(traceback.format_exc())
             equity_curve_s = pd.Series(dtype=float); stats = calculate_statistics(trade_history_df, pd.Series(dtype=float), initial_equity); fig = None
    else:
         equity_curve_s=pd.Series(dtype=float); stats=calculate_statistics(pd.DataFrame(), pd.Series(dtype=float), initial_equity)
         st.write("Aucun trade exécuté ou fermé."); fig = None
    return trade_history_df, equity_curve_s, stats, fig


# ==============================================================
# Fonction Plot Single Trade
# ==============================================================
# (Identique à la version précédente v2.7 - affichera conditionnellement les indicateurs)
def plot_single_trade(data_url, trade_info, params):
    """
    Affiche le graphique détaillé d'un trade spécifique, en ne traçant
    que les indicateurs pertinents pour les paramètres fournis.
    """
    try:
        # Extraction infos trade
        entry_time=trade_info['entry_time']; exit_time=trade_info['exit_time']
        entry_price=trade_info['entry_price']; exit_price=trade_info['exit_price']
        stop_loss=trade_info['stop_loss']; take_profit=trade_info['take_profit']
        trade_type=trade_info['entry_type']

        # Fenêtre de temps et chargement données
        time_buffer = pd.Timedelta(minutes=120)
        plot_start_time = entry_time - time_buffer
        plot_end_time = exit_time + time_buffer
        df_full=pd.read_csv(data_url, parse_dates=['Date'], date_format='%Y-%m-%d %H:%M:%S.%f',
                             encoding='utf-8', on_bad_lines='skip')
        df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
        df_full = df_full.set_index('Date').sort_index()
        plot_df = df_full[(df_full.index >= plot_start_time) & (df_full.index <= plot_end_time)].copy()

        if plot_df.empty: st.warning("Aucune donnée disponible pour la fenêtre de temps du trade."); return None

        # Nettoyage OHLC
        for col in ['Open','High','Low','Close']:
            plot_df[col]=pd.to_numeric(plot_df[col], errors='coerce')
        plot_df.dropna(subset=['Open','High','Low','Close'], inplace=True)
        if plot_df.empty: st.warning("Données OHLC invalides dans la fenêtre de temps."); return None

        # --- Recalcul Indicateurs CONDITIONNEL basé sur params ---
        ema_s_p = params.get('ema_short_period'); ema_l_p = params.get('ema_long_period')
        if ema_s_p is not None and ema_s_p > 0: plot_df['ema_short']=ta.trend.ema_indicator(plot_df['Close'],window=ema_s_p)
        if ema_l_p is not None and ema_l_p > 0: plot_df['ema_long']=ta.trend.ema_indicator(plot_df['Close'],window=ema_l_p)
        reversal_ma_p = params.get('reversal_ma_period')
        if reversal_ma_p is not None and reversal_ma_p > 0: plot_df['reversal_ma'] = ta.trend.sma_indicator(plot_df['Close'], window=reversal_ma_p)
        rsi_p = params.get('rsi_length')
        if rsi_p is not None and rsi_p > 0: plot_df['rsi']=ta.momentum.rsi(plot_df['Close'],window=rsi_p)
        atr_p = params.get('atr_period')
        if atr_p is not None and atr_p > 0: plot_df['atr']=ta.volatility.average_true_range(plot_df['High'],plot_df['Low'],plot_df['Close'],window=atr_p)
        adx_p = params.get('adx_period')
        if adx_p is not None and adx_p > 0:
            if not plot_df[['High', 'Low', 'Close']].isnull().any().any():
                 try: # Essayer pandas_ta d'abord
                     import pandas_ta as pta
                     adx_df = plot_df.ta.adx(length=adx_p); plot_df['adx'] = adx_df[f'ADX_{adx_p}']; plot_df['di_pos'] = adx_df[f'DMP_{adx_p}']; plot_df['di_neg'] = adx_df[f'DMN_{adx_p}']
                 except ImportError: # Fallback sur ta
                     plot_df['adx']=ta.trend.adx(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p); plot_df['di_pos']=ta.trend.adx_pos(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p); plot_df['di_neg']=ta.trend.adx_neg(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p)
            else: plot_df['adx']=pd.NA; plot_df['di_pos']=pd.NA; plot_df['di_neg']=pd.NA

        # Création figure Plotly
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.55, 0.15, 0.15, 0.15], subplot_titles=("Prix & Indicateurs", "RSI", "ATR", "ADX / DI"))

        # --- Trace 1: Prix et MAs (CONDITIONNELLES) ---
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='OHLC'), row=1, col=1)
        if 'ema_short' in plot_df.columns and not plot_df['ema_short'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['ema_short'],mode='lines',name=f'EMA({ema_s_p})',line=dict(color='lightblue',width=1)),row=1,col=1)
        if 'ema_long' in plot_df.columns and not plot_df['ema_long'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['ema_long'],mode='lines',name=f'EMA({ema_l_p})',line=dict(color='yellow',width=1)),row=1,col=1)
        if 'reversal_ma' in plot_df.columns and not plot_df['reversal_ma'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['reversal_ma'],mode='lines',name=f'SMA({reversal_ma_p})',line=dict(color='orange',width=1.5)),row=1,col=1)

        # Lignes Horizontales et Verticales
        lec, ltp, lsl, lex = 'grey', 'lime', 'red', 'fuchsia'; pos_right="bottom right"; pos_left="top right"
        fig.add_hline(y=entry_price, line_dash="dash", line_color=lec, annotation_text="Entrée", annotation_position=pos_right, row=1, col=1)
        fig.add_hline(y=take_profit, line_dash="dot", line_color=ltp, annotation_text="TP", annotation_position=pos_right, row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dot", line_color=lsl, annotation_text="SL", annotation_position=pos_left, row=1, col=1)
        if not pd.isna(exit_price): fig.add_hline(y=exit_price, line_dash="dashdot", line_color=lex, annotation_text="Sortie", annotation_position=pos_left, row=1, col=1)
        lemc="rgba(100,100,255,0.5)";
        fig.add_vline(x=entry_time, line_width=1, line_dash="dash", line_color=lemc)
        if not pd.isna(exit_time): fig.add_vline(x=exit_time, line_width=1, line_dash="dash", line_color=lemc)

        # Marqueurs Entrée/Sortie
        ms = 'triangle-up' if trade_type=='long' else 'triangle-down'; mc = 'lime' if trade_type=='long' else 'red'; mec='fuchsia'
        fig.add_trace(go.Scatter(x=[entry_time], y=[entry_price], mode='markers', name='Entrée Pt', marker=dict(symbol=ms, color=mc, size=12, line=dict(width=1,color='white'))), row=1, col=1)
        if not pd.isna(exit_price) and not pd.isna(exit_time): fig.add_trace(go.Scatter(x=[exit_time], y=[exit_price], mode='markers', name='Sortie Pt', marker=dict(symbol='x', color=mec, size=10, line=dict(width=1,color='white'))), row=1, col=1)

        # --- Trace 2: RSI (CONDITIONNEL) ---
        if 'rsi' in plot_df.columns and not plot_df['rsi'].isnull().all():
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['rsi'], mode='lines', name='RSI', line=dict(color='rgb(180,180,255)', width=1)), row=2, col=1)
            rsi_ob = params.get('rsi_overbought'); rsi_os = params.get('rsi_oversold')
            if rsi_ob is not None: fig.add_hline(y=rsi_ob, line_dash="dash", line_color="red", row=2, col=1)
            if rsi_os is not None: fig.add_hline(y=rsi_os, line_dash="dash", line_color="lime", row=2, col=1)

        # --- Trace 3: ATR (CONDITIONNEL) ---
        if 'atr' in plot_df.columns and not plot_df['atr'].isnull().all():
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['atr'], mode='lines', name='ATR', line=dict(color='darkgrey', width=1)), row=3, col=1)
            atr_thr = params.get('atr_threshold');
            if atr_thr is not None and atr_thr > 0: fig.add_hline(y=atr_thr, line_dash="dot", line_color="cyan", name='Seuil ATR', row=3, col=1)

        # --- Trace 4: ADX / DI (CONDITIONNEL) ---
        if 'adx' in plot_df.columns and not plot_df['adx'].isnull().all():
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['adx'], mode='lines', name='ADX', line=dict(color='white', width=1.5)), row=4, col=1)
            if 'di_pos' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['di_pos'], mode='lines', name='+DI', line=dict(color='green', width=1)), row=4, col=1)
            if 'di_neg' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['di_neg'], mode='lines', name='-DI', line=dict(color='red', width=1)), row=4, col=1)
            adx_thr = params.get('adx_threshold'); # Le seuil ADX spécifique à la strat EMA Cross ou au filtre général
            if adx_thr is not None and adx_thr > 0: fig.add_hline(y=adx_thr, line_dash="dot", line_color="aqua", name='Seuil ADX', row=4, col=1)

        # Mise en forme finale
        fig.update_layout(title=f"Visualisation Trade #{trade_info.name} ({trade_type.upper()})", xaxis_rangeslider_visible=False, height=950, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1), template="plotly_dark")
        fig.update_yaxes(title_text="Prix", row=1, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_yaxes(title_text="RSI", range=[0,100], row=2, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_yaxes(title_text="ATR", row=3, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_yaxes(title_text="ADX/DI", row=4, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_xaxes(gridcolor='rgba(180,180,180,0.3)')

        return fig
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique de trade: {e}")
        st.error(traceback.format_exc()); return None


# ==============================================================
# --- Interface Utilisateur Streamlit ---
# ==============================================================

# --- Barre Latérale (Sidebar) ---
st.sidebar.header("Paramètres du Backtest")
DEFAULT_DATA_URL = "https://lovecrash.online/btc-usd_data_1min.csv"
NETLIFY_DATA_URL = os.environ.get("DATA_URL", DEFAULT_DATA_URL)
st.sidebar.caption(f"Données: {NETLIFY_DATA_URL.split('/')[-1]}")
initial_equity = st.sidebar.number_input("Capital Initial ($)", min_value=1.0, value=10000.0, step=100.0, format="%.2f") # Défaut 10k comme script Pine

# --- Choix de la Stratégie ---
st.sidebar.subheader("Stratégie Principale")
### MODIFICATION START ###
strategy_choice = st.sidebar.radio(
    "Type de Stratégie",
    ('Tendance EMA/RSI + Divergence', 'Reversal (MA + RSI Divergence)', 'Croisement EMA + ADX'), # Ajout nouvelle strat
    index=2, key='strat_choice', # Défaut sur la nouvelle strat
    help="Choisir la logique d'entrée principale.")

strategy_type_arg = 'trend_divergence'
if strategy_choice == 'Reversal (MA + RSI Divergence)': strategy_type_arg = 'reversal_ma_div'
elif strategy_choice == 'Croisement EMA + ADX': strategy_type_arg = 'ema_cross_adx'

is_trend_div_strategy = (strategy_type_arg == 'trend_divergence')
is_reversal_strategy = (strategy_type_arg == 'reversal_ma_div')
is_ema_cross_strategy = (strategy_type_arg == 'ema_cross_adx')
### MODIFICATION END ###

# --- Paramètres des Indicateurs (Affichage Conditionnel) ---
st.sidebar.subheader("Paramètres des Indicateurs")

# Section EMA (Visible si Tendance/Div OU EMA Cross)
if is_trend_div_strategy or is_ema_cross_strategy:
    with st.sidebar.container(border=True):
        st.caption("Params EMA")
        # Toggle seulement pour Tendance/Div
        if is_trend_div_strategy:
             use_ema_rsi = st.toggle("Utiliser Signal EMA/RSI (Tendance/Pullback)", value=True, key="toggle_ema_rsi_trend")
        else: use_ema_rsi = False # Non applicable à EMA Cross

        # Périodes visibles si Tendance/Div ET toggle actif, OU si EMA Cross
        if (is_trend_div_strategy and use_ema_rsi) or is_ema_cross_strategy:
            col_ema1, col_ema2 = st.columns(2)
            with col_ema1:
                ema_s_p=st.number_input("EMA Rapide",min_value=2,value=50,step=1,format="%d", key="ema_s") # Défaut 50 comme script Pine
            with col_ema2:
                ema_l_p=st.number_input("EMA Longue",min_value=2,value=200,step=1,format="%d", key="ema_l") # Défaut 200 comme script Pine
            if ema_l_p <= ema_s_p: st.sidebar.warning("EMA Longue devrait être > EMA Courte.")
        else: # Valeurs par défaut si non affichées
            ema_s_p = 50; ema_l_p = 200
else: # Valeurs par défaut si section non affichée
    use_ema_rsi = False; ema_s_p = 50; ema_l_p = 200

# Section SMA Reversal (Visible seulement si Reversal)
if is_reversal_strategy:
    with st.sidebar.container(border=True):
        st.caption("Params MA Reversal")
        reversal_ma_period_input = st.number_input("Période SMA Reversal", min_value=2, value=100, step=1, format="%d", key="reversal_ma_p")
else: reversal_ma_period_input = 100

# Section RSI (Visible si Tendance/Div ou Reversal)
if is_trend_div_strategy or is_reversal_strategy:
    with st.sidebar.container(border=True):
        st.caption("Params RSI")
        rsi_len_param=st.number_input("Période RSI",min_value=2,max_value=100,value=14,step=1,format="%d",key="rsi_len")
        # Afficher OB/OS seulement si Tendance/Div ET signal EMA/RSI actif
        if is_trend_div_strategy and use_ema_rsi:
            col_rsi1, col_rsi2 = st.columns(2)
            with col_rsi1: rsi_os_val=st.number_input("RSI Oversold",min_value=1,max_value=50,value=30,step=1,format="%d", key="rsi_os")
            with col_rsi2: rsi_ob_val=st.number_input("RSI Overbought",min_value=50,max_value=99,value=70,step=1,format="%d", key="rsi_ob")
            if rsi_ob_val <= rsi_os_val: st.sidebar.error("RSI Overbought doit être > RSI Oversold.")
        else: rsi_os_val = 30; rsi_ob_val = 70
else: # Valeurs par défaut si non pertinent
    rsi_len_param = 14; rsi_os_val = 30; rsi_ob_val = 70

# Section ADX (Visible si EMA Cross OU Filtre ADX Général activé)
use_adx_f = st.sidebar.toggle("Activer Filtre ADX Général", value=False, key="toggle_adx", help="Filtre ADX additionnel (peut s'ajouter à EMA Cross).")
adx_needed_for_ui = is_ema_cross_strategy or use_adx_f
if adx_needed_for_ui:
    with st.sidebar.container(border=True):
         st.caption("Params ADX")
         adx_period_val=st.number_input("Période ADX",min_value=2,max_value=100,value=14,step=1,format="%d", key="adx_p")
         # Seuil Spécifique pour EMA Cross
         if is_ema_cross_strategy:
              adx_threshold_specific_input = st.number_input("Seuil ADX (Strat EMA Cross)",min_value=0.0,max_value=100.0,value=25.0,step=0.1,format="%.1f", key="adx_t_specific", help="Seuil ADX pour valider les croisements EMA.")
         else: adx_threshold_specific_input = 25.0 # Défaut
         # Seuil pour Filtre Général
         if use_adx_f:
              adx_threshold_filter_input = st.number_input("Seuil ADX (Filtre Général)",min_value=0.0,max_value=100.0,value=25.0,step=0.1,format="%.1f", key="adx_t_filter", help="Seuil pour le filtre ADX général (si activé).")
         else: adx_threshold_filter_input = 25.0 # Défaut
else: # Valeurs par défaut si ADX non visible
     adx_period_val=14; adx_threshold_specific_input = 25.0; adx_threshold_filter_input = 25.0

# Section Divergence (Lookback visible si Tendance/Div avec Div OU si Reversal)
use_div_trend_strat = False; use_bull_div = False; use_bear_div = False
if is_trend_div_strategy:
    with st.sidebar.container(border=True):
         st.caption("Options Divergence (Stratégie Tendance/Div)")
         use_div_trend_strat = st.toggle("Activer Signal Divergence RSI", value=False, key="toggle_div_trend")
         if use_div_trend_strat:
              use_bull_div=st.checkbox("Utiliser Div. Haussière (Long)", value=True, key="use_bull_div");
              use_bear_div=st.checkbox("Utiliser Div. Baissière (Short)", value=True, key="use_bear_div")

div_lookback_needed = (is_trend_div_strategy and use_div_trend_strat) or is_reversal_strategy
if div_lookback_needed:
     div_lookback_val = st.number_input("Période Lookback Div RSI", min_value=5, max_value=200, value=30, step=1, format="%d", key='div_lookback_input')
else: div_lookback_val = 30

# --- Section Spécifique Tendance/Divergence (Options supplémentaires) ---
if is_trend_div_strategy:
    st.sidebar.subheader("Options Spécifiques Tendance/Divergence")
    with st.sidebar.container(border=True):
        st.caption("Combinaison & Validité (Tendance/Div)")
        use_points = st.toggle("Utiliser Système de Points", value=False, key="toggle_points")
        pts_ema_rsi_l=1; pts_ema_rsi_s=1; pts_div_bll=1; pts_div_bear=1; score_long_thresh=1; score_short_thresh=1
        if use_points:
            col_pts1, col_pts2 = st.columns(2)
            with col_pts1:
                pts_ema_rsi_l=st.number_input("Pts EMA/RSI Long", min_value=0, value=pts_ema_rsi_l, step=1, format="%d", key="pts_ema_l", disabled=not use_ema_rsi);
                pts_div_bll=st.number_input("Pts Div Bullish", min_value=0, value=pts_div_bll, step=1, format="%d", key="pts_div_l", disabled=not use_div_trend_strat);
                score_long_thresh=st.number_input("Seuil Score Long", min_value=1, value=score_long_thresh, step=1, format="%d", key="thresh_l")
            with col_pts2:
                pts_ema_rsi_s=st.number_input("Pts EMA/RSI Short", min_value=0, value=pts_ema_rsi_s, step=1, format="%d", key="pts_ema_s", disabled=not use_ema_rsi);
                pts_div_bear=st.number_input("Pts Div Bearish", min_value=0, value=pts_div_bear, step=1, format="%d", key="pts_div_s", disabled=not use_div_trend_strat);
                score_short_thresh=st.number_input("Seuil Score Short", min_value=1, value=score_short_thresh, step=1, format="%d", key="thresh_s")

        signal_validity_input = st.number_input("Validité Signal (barres)", min_value=1, value=1, step=1, format="%d", key='sig_validity')
else: use_points = False; signal_validity_input = 1; pts_ema_rsi_l=1; pts_ema_rsi_s=1; pts_div_bll=1; pts_div_bear=1; score_long_thresh=1; score_short_thresh=1

# --- Section Risque/Sortie ---
st.sidebar.subheader("Gestion du Risque et Sortie")

# Sizing (Conditionnel pour EMA Cross ADX)
if is_ema_cross_strategy:
     sizing_type_arg = 'risk_pct' # Forcer pour cette stratégie
     st.sidebar.write("Sizing: Risque % (forcé pour cette strat)")
     risk_pct_input = st.sidebar.number_input("Risque par Trade (%)", min_value=0.01, max_value=5.0, value=0.5, step=0.01, format="%.2f", key="risk_pct_ema_cross") # Défaut 0.5% comme script Pine
     risk_pct_val = risk_pct_input / 100.0
     fixed_lot_val = 0.01 # Non utilisé mais besoin d'une valeur
else: # Choix possible pour les autres strats
     sizing_mode_input = st.sidebar.radio( "Méthode Sizing", ["Risque %", "Lot Fixe"], index=0, key='sizing_mode')
     risk_pct_val = 0.0; fixed_lot_val = 0.0; risk_pct_input=0.5
     if sizing_mode_input == "Risque %":
          risk_pct_input = st.sidebar.number_input("Risque par Trade (%)", min_value=0.01, max_value=5.0, value=0.5, step=0.01, format="%.2f")
          sizing_type_arg = "risk_pct"; risk_pct_val = risk_pct_input / 100.0; fixed_lot_val = 0.01
     else:
          fixed_lot_size_input = st.sidebar.number_input("Taille Lot Fixe", min_value=0.0001, value=0.01, step=0.001, format="%.4f")
          sizing_type_arg = "fixed_lot"; fixed_lot_val = fixed_lot_size_input; risk_pct_val = 0.005

# Stop Loss (Conditionnel pour EMA Cross ADX)
if is_ema_cross_strategy:
     sl_type_arg = 'percentage' # Forcer pour cette stratégie
     st.sidebar.write("Stop Loss: % Prix (forcé pour cette strat)")
     sl_pct_input = st.sidebar.number_input("Stop Loss (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f", key="sl_pct_ema_cross") # Défaut 0.5% comme script Pine
     sl_pct = sl_pct_input / 100.0
     atr_multiplier_sl_input = 1.5 # Non utilisé mais besoin d'une valeur
else: # Choix possible pour les autres strats
     sl_mode_input = st.sidebar.radio("Type Stop Loss", ["% Prix", "ATR"], index=1, key='sl_mode')
     sl_pct = 0.0; atr_multiplier_sl_input = 1.5
     if sl_mode_input == "% Prix":
          sl_pct_input = st.sidebar.number_input("Stop Loss (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f")
          sl_type_arg = "percentage"; sl_pct = sl_pct_input / 100.0
     else: # ATR
          atr_needed_for_sl = True # Si on arrive ici, ATR est nécessaire
          if 'atr_period_val' not in locals(): # S'assurer que la période ATR est définie (si section masquée)
               atr_period_val = st.number_input("Période ATR (pour SL)", min_value=2, max_value=100, value=14, step=1, format="%d", key="atr_p_sl_only")
          atr_multiplier_sl_input = st.sidebar.number_input("Multiplicateur ATR (SL)", min_value=0.1, max_value=10.0, value=1.5, step=0.1, format="%.1f")
          sl_type_arg = "atr"

# Take Profit (Toujours visible, basé sur RR)
tp_mult = st.sidebar.number_input("Ratio Risque/Rendement (RR)", min_value=0.1, max_value=20.0, value=4.0, step=0.1, format="%.1f", key="rr_ratio", help="TP = Prix Entrée +/- (RR * Distance_SL). Défaut 4.0 comme script Pine.") # Défaut 4.0

# Filtre ATR Min (Visible seulement si ATR pertinent pour SL ou filtre)
atr_needed_for_filter = sl_type_arg == 'ATR' # Vérifier si SL ATR est le type choisi
if not atr_needed_for_filter: # Si SL n'est pas ATR, vérifier si l'input existe
    atr_filter_threshold = st.sidebar.number_input("ATR Min. pour Trader", min_value=0.0, value=0.0, step=0.01, format="%.5f", help="Si > 0, ignore signaux si ATR < seuil.", key="atr_filt_thresh_alt")
    if atr_filter_threshold > 0 and 'atr_period_val' not in locals(): # S'assurer que la période est définie
         atr_period_val = st.number_input("Période ATR (pour Filtre)", min_value=2, max_value=100, value=14, step=1, format="%d", key="atr_p_filt_only")
elif 'atr_filter_threshold' not in locals(): # S'assurer que la variable existe si SL est ATR mais filtre pas encore créé
    atr_filter_threshold = st.sidebar.number_input("ATR Min. pour Trader", min_value=0.0, value=0.0, step=0.01, format="%.5f", help="Si > 0, ignore signaux si ATR < seuil.", key="atr_filt_thresh_main")


# --- Filtres Additionnels & Concurrence ---
st.sidebar.subheader("Filtres Additionnels et Concurrence")
# Filtre Min Profit Points
use_min_profit_filter_input = st.sidebar.toggle( "Activer Filtre Min Profit Points", value=False, key="toggle_min_profit")
min_profit_points_input = 0.0
if use_min_profit_filter_input: min_profit_points_input = st.sidebar.number_input( "Seuil Min Profit Points (Distance Entry->TP)", min_value=0.0, value=50.0, step=1.0, format="%.5f", key="min_profit_val")
# Concurrence
one_trade_at_a_time_input = st.sidebar.checkbox("Limiter à un seul trade ouvert", value=True) # Défaut True comme script Pine

# --- Bouton de Lancement ---
params_valid = True
if (is_trend_div_strategy or is_ema_cross_strategy) and ema_l_p <= ema_s_p: params_valid = False
# Autres validations si besoin

run_button = st.sidebar.button("🚀 Lancer le Backtest", disabled=not params_valid, use_container_width=True)
st.sidebar.markdown("---"); st.sidebar.info("Backtester v2.8") # Version mise à jour

# --- Zone d'Affichage Principale ---
st.header("Résultats du Backtest")
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated=False; st.session_state.trade_history=pd.DataFrame(); st.session_state.equity_curve=pd.Series(dtype=float)
    st.session_state.statistics={}; st.session_state.equity_fig=None; st.session_state.backtest_params={}

if run_button:
    st.session_state.results_calculated = False; st.session_state.trade_history=pd.DataFrame(); st.session_state.equity_curve=pd.Series(dtype=float)
    st.session_state.statistics={}; st.session_state.equity_fig=None

    # --- Détermination Indicateurs à Calculer (Strict) ---
    calc_ema = is_trend_div_strategy or is_ema_cross_strategy
    calc_reversal_ma = is_reversal_strategy
    calc_rsi = is_trend_div_strategy or is_reversal_strategy # Pas besoin pour EMA Cross
    calc_divergence = (is_trend_div_strategy and use_div_trend_strat) or is_reversal_strategy
    # ATR nécessaire si SL ATR OU filtre ATR actif
    atr_needed = sl_type_arg == 'atr' or atr_filter_threshold > 0
    calc_atr = atr_needed
    # ADX nécessaire si strat EMA Cross OU filtre général ADX actif
    calc_adx = is_ema_cross_strategy or use_adx_f

    # Prépare infos pour affichage
    strat_display_name = ""; filter_info = [];
    if use_adx_f: filter_info.append(f"Filtre ADX({adx_period_val})>{adx_threshold_filter_input}")
    if atr_filter_threshold > 0: filter_info.append(f"Filtre ATR({atr_period_val})>{atr_filter_threshold:.5f}")
    if use_min_profit_filter_input: filter_info.append(f"Filtre MinPtsTP>{min_profit_points_input:.5f}")
    filter_str = ", ".join(filter_info) if filter_info else "Aucun"

    if strategy_type_arg == 'trend_divergence':
        # ... (affichage Tendance/Div identique à v2.7) ...
        active_signals = []
        if use_ema_rsi: active_signals.append(f"EMA({ema_s_p}/{ema_l_p})/RSI({rsi_len_param})")
        if use_div_trend_strat:
             div_sig_parts = [];
             if use_bull_div: div_sig_parts.append("Bull")
             if use_bear_div: div_sig_parts.append("Bear")
             if div_sig_parts: active_signals.append(f"Div({'/'.join(div_sig_parts)})({div_lookback_val})")
        signal_info = "+".join(active_signals) if active_signals else "AUCUN"
        combo_logic = "Points" if use_points else "OU Simple"; sig_valid_info = f"(Validité: {signal_validity_input}b)"
        strat_display_name = f"Tendance/Div [{signal_info}]/{combo_logic}{sig_valid_info}"
    elif strategy_type_arg == 'reversal_ma_div':
        strat_display_name = f"Reversal [SMA({reversal_ma_period_input})/RSI Div({div_lookback_val})]"
    elif strategy_type_arg == 'ema_cross_adx':
        strat_display_name = f"EMA Cross({ema_s_p}/{ema_l_p}) + ADX({adx_period_val})>{adx_threshold_specific_input}"

    concurrency_mode = "Unique" if one_trade_at_a_time_input else "Multiple"
    sizing_info = f"{risk_pct_input:.2f}% Risk" # Forcé pour EMA Cross, sinon dépend de l'UI
    sl_info = f"{sl_pct_input:.2f}% Px" # Forcé pour EMA Cross, sinon dépend de l'UI
    info_str = f"Lancement: {strat_display_name} | Sizing:{sizing_info} | SL:{sl_info} | RR:{tp_mult:.1f} | Filtres:[{filter_str}] | Conc:{concurrency_mode}"
    st.info(info_str); progress_placeholder_area = st.empty()

    # Stocke params utilisés pour visu trade (filtrés)
    current_params = {
        "ema_short_period": ema_s_p if calc_ema else None,
        "ema_long_period": ema_l_p if calc_ema else None,
        "reversal_ma_period": reversal_ma_period_input if calc_reversal_ma else None,
        "rsi_length": rsi_len_param if calc_rsi else None,
        "rsi_oversold": rsi_os_val if calc_rsi and is_trend_div_strategy and use_ema_rsi else None,
        "rsi_overbought": rsi_ob_val if calc_rsi and is_trend_div_strategy and use_ema_rsi else None,
        "atr_period": atr_period_val if calc_atr else None,
        "atr_threshold": atr_filter_threshold if atr_filter_threshold > 0 else None,
        "div_lookback_period": div_lookback_val if calc_divergence else None,
        "adx_period": adx_period_val if calc_adx else None,
        # Stocker le seuil ADX réellement utilisé (soit spécifique, soit filtre, soit aucun)
        "adx_threshold": adx_threshold_specific_input if is_ema_cross_strategy else (adx_threshold_filter_input if use_adx_f else None),
    }
    st.session_state.backtest_params = current_params

    st.write("Préparation données (via cache)...")
    df_preprocessed = load_data_and_indicators(
        url=NETLIFY_DATA_URL,
        calc_ema=calc_ema, short_ema_p=ema_s_p, long_ema_p=ema_l_p,
        calc_rsi=calc_rsi, rsi_p=rsi_len_param,
        calc_atr=calc_atr, atr_p=atr_period_val,
        calc_divergence=calc_divergence, div_lookback_p=div_lookback_val,
        calc_adx=calc_adx, adx_p=adx_period_val,
        calc_reversal_ma=calc_reversal_ma, reversal_ma_p=reversal_ma_period_input
        )
    st.write("Données prêtes pour backtest.")

    if not df_preprocessed.empty:
        with st.spinner("Backtest en cours..."):
            # Récupérer les bonnes valeurs pour les arguments de backtest_strategy
            final_risk_pct = risk_pct_val # Déjà calculé en décimal
            final_sl_pct = sl_pct       # Déjà calculé en décimal
            final_tp_mult = tp_mult     # Nom différent (RR)

            th, ec, stats, efig = backtest_strategy(
                df_processed=df_preprocessed, initial_equity=initial_equity,
                strategy_type=strategy_type_arg,
                # Risque/Sizing/SL/TP
                risk_percentage=final_risk_pct, sizing_type=sizing_type_arg, fixed_lot_size=fixed_lot_val,
                sl_type=sl_type_arg, stop_loss_percentage=final_sl_pct, atr_multiplier_sl=atr_multiplier_sl_input,
                take_profit_multiplier=final_tp_mult,
                # Filtres / Concurrence
                atr_threshold=atr_filter_threshold, one_trade_at_a_time=one_trade_at_a_time_input,
                use_adx_filter=use_adx_f, adx_threshold=adx_threshold_filter_input, # Seuil filtre général
                use_min_profit_points_filter=use_min_profit_filter_input, min_profit_points_threshold=min_profit_points_input,
                # Params Indicateurs (pour info et si besoin interne)
                ema_short_period=ema_s_p, ema_long_period=ema_l_p, rsi_length=rsi_len_param,
                rsi_oversold=rsi_os_val, rsi_overbought=rsi_ob_val, atr_period=atr_period_val, adx_period=adx_period_val,
                reversal_ma_period=reversal_ma_period_input, div_lookback_period=div_lookback_val,
                # adx_threshold=adx_threshold_specific_input, # Passer le seuil spécifique pour EMA Cross ici
                 # --- Params spécifiques Tendance/Div ---
                use_ema_rsi_signal=use_ema_rsi, use_bullish_div=use_bull_div, use_bearish_div=use_bear_div,
                use_points_system=use_points,
                points_ema_rsi_long=pts_ema_rsi_l, points_ema_rsi_short=pts_ema_rsi_s, points_bull_div=pts_div_bll,
                points_bear_div=pts_div_bear, long_score_threshold=score_long_thresh, short_score_threshold=score_short_thresh,
                signal_validity_bars=signal_validity_input,
                progress_placeholder=progress_placeholder_area)
        progress_placeholder_area.empty()

        st.session_state.trade_history=th; st.session_state.equity_curve=ec
        st.session_state.statistics=stats; st.session_state.equity_fig=efig
        st.session_state.results_calculated=True
        st.success("Backtest terminé !")
    else:
        st.error("Chargement ou préparation des données échoué, backtest annulé.")
        st.session_state.results_calculated = False

# --- Affichage des Résultats ---
if st.session_state.results_calculated:
    stats=st.session_state.statistics
    equity_fig=st.session_state.equity_fig
    trade_history=st.session_state.trade_history

    if stats and isinstance(stats, dict):
        st.subheader("Période de Trading");
        first_date=stats.get('First Trade Date'); last_date=stats.get('Last Trade Date');
        date_format='%Y-%m-%d %H:%M:%S';
        date_col1, date_col2 = st.columns(2)
        with date_col1: st.markdown(f"**Premier Trade:**"); st.write(first_date.strftime(date_format) if pd.notna(first_date) else "N/A")
        with date_col2: st.markdown(f"**Dernier Trade:**"); st.write(last_date.strftime(date_format) if pd.notna(last_date) else "N/A");
        st.divider()

        st.subheader("Statistiques Clés");
        col1,col2,col3=st.columns(3);
        col1.metric("Profit ($)",f"{stats.get('Total Profit',0):,.2f}");
        col2.metric("Profit (%)",f"{stats.get('Profit (%)',0):.2f}%");
        pf_val = stats.get('Profit Factor', 0)
        col3.metric("PF",f"{pf_val:.2f}" if pf_val != float('inf') else "Inf")

        col4,col5,col6=st.columns(3);
        col4.metric("Trades",f"{stats.get('Number of Trades',0):,}");
        col5.metric("Win (%)",f"{stats.get('Winning Trades (%)',0):.2f}%");
        col6.metric("Max DD (%)",f"{stats.get('Max Drawdown (%)',0):.2f}%")

        col7,col8,col9=st.columns(3);
        col7.metric("Max Loss Str",f"{stats.get('Max Consecutive Losing Trades',0)}");
        col8.metric("Avg Loss Str",f"{stats.get('Average Consecutive Losing Trades',0):.1f}");
        col9.metric("Final Eq ($)",f"{stats.get('Final Equity', initial_equity):,.2f}");
        st.divider()

        st.subheader("Courbe d'Équité");
        if equity_fig:
            st.pyplot(equity_fig)
            plt.close(equity_fig)
        elif stats.get('Number of Trades', 0) > 0 :
            st.warning("Graphique d'équité non généré (erreur possible durant le tracé ou stats manquantes).")
        else:
            st.info("Aucun trade exécuté, pas de courbe d'équité.")

        st.subheader("Historique des Trades")
        if not trade_history.empty:
            st.dataframe(pd.concat([trade_history.head(),trade_history.tail()]).style.format(
                 {"entry_price":"{:.5f}", "exit_price":"{:.5f}", "stop_loss":"{:.5f}",
                  "take_profit":"{:.5f}", "profit":"{:,.2f}", "size":"{:.4f}"}
            ))
            csv_data = trade_history.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Télécharger l'historique complet (CSV)",
                               data=csv_data, file_name='trade_history.csv', mime='text/csv')
        elif stats.get('Number of Trades',-1)==0:
            st.info("Aucun trade exécuté.")
        st.divider()

        st.subheader("Visualisation d'un Trade Spécifique")
        if not trade_history.empty:
            max_trade_index = len(trade_history) - 1;
            trade_ids = trade_history.index.tolist()
            default_idx = 0 if max_trade_index >= 0 else None
            if default_idx is not None:
                 selected_trade_id_idx = st.selectbox(f"Choisir l'index du trade à visualiser (0 à {max_trade_index})",
                                                  options=trade_ids, index=default_idx, key='trade_selector_idx')

                 if selected_trade_id_idx is not None and st.button("Afficher Graphique Trade", key='show_trade_btn'):
                      with st.spinner("Génération graphique du trade..."):
                          # Récupérer la ligne par index (loc peut être ambigu si index non unique/reset)
                          trade_details = trade_history.iloc[selected_trade_id_idx]
                          backtest_params_for_plot = st.session_state.backtest_params
                          single_trade_fig = plot_single_trade(NETLIFY_DATA_URL, trade_details, backtest_params_for_plot)
                          if single_trade_fig: st.plotly_chart(single_trade_fig, use_container_width=True)
                          else: st.warning("Impossible d'afficher le graphique pour ce trade.")
            else: st.info("Aucun trade disponible pour la sélection.")
        else: st.info("Aucun trade dans l'historique à visualiser.")

    elif not st.session_state.trade_history.empty:
        st.error("Erreur lors de la récupération ou du calcul des statistiques, mais historique de trades disponible.")
        st.dataframe(st.session_state.trade_history)
    elif not params_valid:
         st.warning("Certains paramètres sont invalides. Veuillez corriger les erreurs dans la barre latérale.")
    else:
         st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer le Backtest'.")
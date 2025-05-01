# -*- coding: utf-8 -*-
# app.py (v3.4.1 - Code complet avec date corrigée, filtre ADX, rentabilité périodique, graphique explicatif, sélection dates)

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
import numpy as np
import datetime # Importation nécessaire pour les dates

# Configuration Matplotlib et Page Streamlit
matplotlib.use('Agg')
st.set_page_config(layout="wide", page_title="Backtester - Stratégie Main v3.4.1") # Version mise à jour

# --- Injection CSS et Titre ---
# (CSS identique - omis)
font_path = "PASTOROF.ttf"; font_css = ""
try:
    if os.path.exists(font_path):
        font_css = f"""<style> @font-face {{ font-family: 'PASTOROF'; src: url('{font_path}') format('truetype'); }} h1[data-testid="stHeading"], .stApp > header h1 {{ font-family: 'PASTOROF', sans-serif !important; font-size: 3rem !important; }} body {{ font-family: 'Helvetica Neue', sans-serif; }} div.main {{ background-color: #000000 !important; color: #FFFFFF !important; }} div[data-testid="stAppViewContainer"], .stApp {{ color: #FFFFFF !important; }} div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label, div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label, div[data-testid="stToggle"] label, div[data-testid="stSelectbox"] label {{ color: #EEEEEE !important; }} div[data-testid="stMetric"] {{ background-color: #111111 !important; border-radius: 5px; padding: 10px; color: #FFFFFF !important; }} div[data-testid="stMetric"] > label {{ color: #AAAAAA !important; }} div[data-testid="stMetric"] > div:nth-of-type(2) {{ color: #FFFFFF !important; }} .stDataFrame {{ color: #333; }} </style>"""
    else:
        font_css = """<style> h1[data-testid="stHeading"], .stApp > header h1 { font-size: 3rem !important; } body { font-family: 'Helvetica Neue', sans-serif; } div.main { background-color: #000000 !important; color: #FFFFFF !important; } div[data-testid="stAppViewContainer"], .stApp { color: #FFFFFF !important; } div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label, div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label, div[data-testid="stToggle"] label, div[data-testid="stSelectbox"] label { color: #EEEEEE !important; } div[data-testid="stMetric"] { background-color: #111111 !important; border-radius: 5px; padding: 10px; color: #FFFFFF !important; } div[data-testid="stMetric"] > label { color: #AAAAAA !important; } div[data-testid="stMetric"] > div:nth-of-type(2) { color: #FFFFFF !important; } .stDataFrame { color: #333; } </style>"""
    st.markdown(font_css, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Police perso/CSS non chargé: {e}", icon="🎨")
st.title("Backtester - Stratégie Main")
# --- Fin Style ---

# ==============================================================
# Fonction de Calcul des Statistiques (Identique v2.9.1)
# ==============================================================
def calculate_statistics(trade_history, equity_curve, initial_equity, equity_history_list):
    """Calcule les statistiques clés du backtest."""
    stats = {}
    number_of_trades = len(trade_history)
    stats['Number of Trades'] = number_of_trades
    stats['First Trade Date'] = None; stats['Last Trade Date'] = None
    stats['Total Profit'] = 0; stats['Final Equity'] = initial_equity
    stats['Profit (%)'] = 0; stats['Winning Trades (%)'] = 0
    stats['Max Drawdown (%)'] = 0; stats['Max Consecutive Losing Trades'] = 0
    stats['Average Consecutive Losing Trades'] = 0; stats['Average Profit per Trade'] = 0
    stats['Profit Factor'] = 0

    if number_of_trades == 0:
        stats['Final Equity'] = initial_equity
        stats['Max Drawdown (%)'] = 0
        return stats

    if not isinstance(trade_history, pd.DataFrame):
        trade_history_df = pd.DataFrame(trade_history)
    else:
        trade_history_df = trade_history

    if trade_history_df.empty:
        stats['Final Equity'] = initial_equity
        stats['Max Drawdown (%)'] = 0
        return stats

    try:
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

        if equity_history_list:
            equity_values = equity_history_list
            equity_curve_for_dd = pd.Series(equity_values)
            max_drawdown = 0
            peak = equity_curve_for_dd[0]
            for equity_val in equity_curve_for_dd:
                if equity_val > peak:
                    peak = equity_val
                drawdown = (peak - equity_val) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            stats['Max Drawdown (%)'] = max_drawdown * 100
        else:
             stats['Max Drawdown (%)'] = 0

        losing_streak, max_losing_streak, losing_streak_lengths = 0, 0, []
        for profit in trade_history_df_sorted['profit']:
            if profit <= 0:
                losing_streak += 1
            else:
                if losing_streak > 0: losing_streak_lengths.append(losing_streak)
                max_losing_streak = max(max_losing_streak, losing_streak); losing_streak = 0
        if losing_streak > 0:
            losing_streak_lengths.append(losing_streak); max_losing_streak = max(max_losing_streak, losing_streak)

        stats['Max Consecutive Losing Trades'] = max_losing_streak
        stats['Average Consecutive Losing Trades'] = sum(losing_streak_lengths) / len(losing_streak_lengths) if losing_streak_lengths else 0
        stats['Average Profit per Trade'] = total_profit / number_of_trades if number_of_trades > 0 else 0

        gross_profit = winning_trades['profit'].sum(); gross_loss = abs(losing_trades['profit'].sum())
        stats['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    except Exception as e:
        st.error(f"Erreur calcul statistiques: {e}"); st.error(traceback.format_exc())
        default_stats = {k: 0 for k in ['Number of Trades', 'Total Profit', 'Profit (%)', 'Winning Trades (%)', 'Max Drawdown (%)', 'Max Consecutive Losing Trades', 'Average Consecutive Losing Trades', 'Average Profit per Trade', 'Profit Factor']}
        default_stats['Final Equity'] = initial_equity
        return default_stats

    return stats


# ==============================================================
# Fonction Chargement Données & Indicateurs (Identique v3.2)
# ==============================================================
@st.cache_data
def load_data_and_indicators(file_path,
                             pine_ema1_p=1000, pine_ema2_p=5000,
                             rsi_p=14,
                             calc_adx=False, adx_p=14):
    """ Charge données depuis fichier CSV local et calcule indicateurs (EMA, RSI, option ADX)."""
    active_calcs = [f"PineEMA({pine_ema1_p},{pine_ema2_p})", f"RSI({rsi_p})"]
    if calc_adx: active_calcs.append(f"ADX({adx_p})")
    st.write(f"CACHE MISS: Chargement/Prep Données depuis '{os.path.basename(file_path)}' ({', '.join(active_calcs)})...")
    calculated_indic_cols = []
    try:
        if not os.path.exists(file_path):
            st.error(f"Erreur: Le fichier '{file_path}' est introuvable.")
            return pd.DataFrame()

        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        date_column_name = 'Open time'

        if date_column_name not in df.columns:
             st.error(f"Erreur critique: Colonne '{date_column_name}' non trouvée dans {os.path.basename(file_path)}.")
             return pd.DataFrame()

        try:
            df['timestamp'] = pd.to_datetime(df[date_column_name], errors='coerce')
            if df['timestamp'].isnull().sum() > len(df) * 0.9:
                st.error(f"Échec parsing date colonne '{date_column_name}'.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Erreur parsing date colonne '{date_column_name}': {e}")
            return pd.DataFrame()

        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        numeric_cols=['Open','High','Low','Close']

        for col in numeric_cols:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                 st.error(f"Colonne numérique attendue '{col}' non trouvée.")
                 return pd.DataFrame()
        df.dropna(subset=numeric_cols, inplace=True)

        current_rows = len(df)
        min_rows_needed = 1
        min_rows_needed = max(min_rows_needed, pine_ema1_p + 1, pine_ema2_p + 1)
        min_rows_needed = max(min_rows_needed, rsi_p + 1)
        if calc_adx: min_rows_needed = max(min_rows_needed, adx_p * 2)

        if current_rows < min_rows_needed: st.error(f"Données insuffisantes ({current_rows} lignes valides) pour calculer indicateurs (min requis: ~{min_rows_needed})."); return pd.DataFrame()

        if pine_ema1_p < len(df): df['pine_ema1']=ta.trend.ema_indicator(df['Close'],window=pine_ema1_p); calculated_indic_cols.append('pine_ema1')
        if pine_ema2_p < len(df): df['pine_ema2']=ta.trend.ema_indicator(df['Close'],window=pine_ema2_p); calculated_indic_cols.append('pine_ema2')

        if rsi_p < len(df): df['rsi']=ta.momentum.rsi(df['Close'],window=rsi_p); calculated_indic_cols.append('rsi')
        else: df['rsi'] = pd.NA

        if calc_adx and adx_p*2 < len(df):
            if df[['High', 'Low', 'Close']].isnull().any().any():
                 st.warning("NaNs dans OHLC avant calcul ADX."); df['adx']=pd.NA; df['di_pos']=pd.NA; df['di_neg']=pd.NA
            else:
                try: import pandas_ta as pta; adx_df = df.ta.adx(length=adx_p); df['adx'] = adx_df[f'ADX_{adx_p}']; df['di_pos'] = adx_df[f'DMP_{adx_p}']; df['di_neg'] = adx_df[f'DMN_{adx_p}']
                except ImportError: st.info("Utilisation de 'ta' pour ADX (pandas_ta recommandé)."); df['adx']=ta.trend.adx(df['High'],df['Low'],df['Close'],window=adx_p); df['di_pos']=ta.trend.adx_pos(df['High'],df['Low'],df['Close'],window=adx_p); df['di_neg']=ta.trend.adx_neg(df['High'],df['Low'],df['Close'],window=adx_p)
                calculated_indic_cols.extend(['adx', 'di_pos', 'di_neg'])
        else: df['adx']=pd.NA; df['di_pos']=pd.NA; df['di_neg']=pd.NA

        cols_to_check_for_na = [col for col in calculated_indic_cols if col in df.columns]
        if cols_to_check_for_na:
            first_valid_index = df[cols_to_check_for_na].first_valid_index()
            if first_valid_index is not None:
                 df = df.loc[first_valid_index:]
            else:
                 st.error("Aucune ligne valide après calcul des indicateurs.")
                 return pd.DataFrame()

        st.write("CACHE MISS: Fin chargement et préparation.")
        return df
    except FileNotFoundError:
        st.error(f"Erreur critique: Le fichier '{file_path}' n'a pas été trouvé.")
        return pd.DataFrame()
    except Exception as e: st.error(f"Erreur irrécupérable chargement/préparation données depuis '{os.path.basename(file_path)}': {e}"); st.error(traceback.format_exc()); return pd.DataFrame()


# ==============================================================
# Fonction Principale de Backtesting (Corrigée v3.4.1 - SyntaxError)
# ==============================================================
def backtest_strategy(df_processed, initial_equity=5000,
                      # Params Pine Script "Main"
                      pine_ema1_period=1000, pine_ema2_period=5000,
                      rsi_length=14, rsi_oversold=30, rsi_overbought=70,
                      pine_sl_percentage=0.002,
                      # Sizing
                      sizing_type='risk_pct',
                      pine_risk_percentage=0.005,
                      pine_fixed_lot_size=0.01,
                      # TP
                      pine_tp_multiplier=5.0,
                      # Filtres/Options communes
                      use_adx_filter=False, adx_threshold=25.0,
                      one_trade_at_a_time=True,
                      use_min_profit_points_filter=False,
                      min_profit_points_threshold=0.0,
                      progress_placeholder=None ):
    """ Effectue le backtest pour la stratégie PineScript Main avec filtre ADX optionnel. """
    if df_processed.empty: st.error("Impossible lancer backtest: DataFrame vide."); return pd.DataFrame(), pd.Series(dtype=float), {}, None, []

    fig = None; closed_trades_history = []; equity_history = [initial_equity]; equity = initial_equity; open_positions = []; trade_id_counter = 0; total_rows = len(df_processed)
    st.write(f"Début boucle backtesting ({total_rows} bougies) - Stratégie: PineScript Main")

    required_cols = ['Open', 'High', 'Low', 'Close', 'pine_ema1', 'pine_ema2', 'rsi']
    if use_adx_filter: required_cols.extend(['adx'])

    missing_cols = [col for col in set(required_cols) if col not in df_processed.columns or df_processed[col].isnull().all()]
    if missing_cols: st.error(f"Erreur: Colonnes critiques manquantes/vides : {', '.join(missing_cols)}."); return pd.DataFrame(), pd.Series(dtype=float), {}, None, []

    for i, (index, row) in enumerate(df_processed.iterrows()):
        if progress_placeholder and (i % 500 == 0 or i == total_rows - 1): prog = float(i+1)/total_rows; perc = min(int(prog*100),100); progress_placeholder.text(f"Progression: {perc}%")

        signal_price=row['Close']; current_high=row['High']; current_low=row['Low']; adx_val=row.get('adx', 0);
        if pd.isna(signal_price) or signal_price <= 0: continue

        positions_to_remove = []
        for position in open_positions:
            exit_price = None; pos_id=position['id']; pos_entry_price=position['entry_price']; pos_type=position['type']; pos_sl=position['stop_loss']; pos_tp=position['take_profit']; pos_size=position['size']; pos_entry_time=position['entry_time']
            if pos_type=='long':
                if current_low <= pos_sl: exit_price = pos_sl
                elif current_high >= pos_tp: exit_price = pos_tp
            elif pos_type=='short':
                if current_high >= pos_sl: exit_price = pos_sl
                elif current_low <= pos_tp: exit_price = pos_tp

            if exit_price is not None:
                if pos_type=='long': profit = (exit_price - pos_entry_price) * pos_size
                else: profit = (pos_entry_price - exit_price) * pos_size
                equity += profit; equity = max(equity, 0)
                closed_trades_history.append({'trade_id':pos_id, 'entry_time':pos_entry_time, 'entry_price':pos_entry_price, 'entry_type':pos_type, 'size':pos_size, 'stop_loss':pos_sl, 'take_profit':pos_tp, 'exit_time':index, 'exit_price':exit_price, 'profit':profit})
                equity_history.append(equity); positions_to_remove.append(position)

        for closed_pos in positions_to_remove: open_positions.remove(closed_pos)

        if use_adx_filter and (pd.isna(adx_val) or adx_val < adx_threshold):
             continue

        final_long_signal = False; final_short_signal = False
        ema1 = row.get('pine_ema1')
        ema2 = row.get('pine_ema2')
        rsi_val = row.get('rsi')

        if not pd.isna(signal_price) and not pd.isna(ema1) and not pd.isna(ema2) and not pd.isna(rsi_val):
            long_cond = signal_price > ema1 and ema1 > ema2 and rsi_val < rsi_oversold
            short_cond = signal_price < ema1 and ema1 < ema2 and rsi_val > rsi_overbought
            if long_cond: final_long_signal = True
            elif short_cond: final_short_signal = True

        can_enter = True if not one_trade_at_a_time else (len(open_positions) == 0)
        if can_enter and (final_long_signal or final_short_signal):
            is_long = final_long_signal; is_short = final_short_signal and not is_long
            if is_long or is_short:
                if equity <= 0: st.warning("Équité <= 0. Arrêt."); break

                actual_entry_price, stop_loss_price, take_profit_price, risk_per_unit, position_size = None, None, None, 0.0, 0.0
                actual_entry_price = signal_price

                if actual_entry_price is None or pd.isna(actual_entry_price) or actual_entry_price <= 0: continue

                try:
                    stop_loss_price = actual_entry_price * (1 - pine_sl_percentage if is_long else 1 + pine_sl_percentage)
                    if (is_long and stop_loss_price >= actual_entry_price) or (is_short and stop_loss_price <= actual_entry_price): continue
                    risk_per_unit = abs(actual_entry_price - stop_loss_price)
                    if risk_per_unit <= 1e-9: continue
                    tp_offset = risk_per_unit * pine_tp_multiplier
                    take_profit_price = actual_entry_price + tp_offset if is_long else actual_entry_price - tp_offset
                    if (is_long and take_profit_price <= actual_entry_price) or (is_short and take_profit_price >= actual_entry_price): continue

                    if sizing_type == 'risk_pct':
                        risk_amount = equity * pine_risk_percentage
                        position_size = risk_amount / risk_per_unit if risk_per_unit > 1e-9 and equity > 0 else 0.0
                    elif sizing_type == 'fixed_lot':
                        position_size = pine_fixed_lot_size
                    else: continue
                except Exception as e: continue

                execute_this_trade = True
                if use_min_profit_points_filter and take_profit_price is not None:
                    potential_profit_points = abs(take_profit_price - actual_entry_price)
                    if potential_profit_points < min_profit_points_threshold: execute_this_trade = False

                if execute_this_trade and position_size > 1e-9 and stop_loss_price is not None and take_profit_price is not None:
                    trade_id_counter += 1
                    new_position = {
                        'id': trade_id_counter,
                        'entry_time': index,
                        'entry_price': actual_entry_price,
                        'type': 'long' if is_long else 'short',
                        'size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price
                    }
                    open_positions.append(new_position)

    st.write("Fin boucle backtesting.")
    trade_history_df=pd.DataFrame(closed_trades_history)
    equity_curve_s = pd.Series(dtype=float); fig = None

    if equity_history:
        equity_dates = [df_processed.index[0]] if not df_processed.empty else []
        if not trade_history_df.empty:
            equity_dates.extend(pd.to_datetime(trade_history_df['exit_time']).tolist())
        min_len = min(len(equity_history), len(equity_dates))
        temp_equity_curve = pd.Series(equity_history[:min_len], index=pd.to_datetime(equity_dates[:min_len]))
        equity_curve_s = temp_equity_curve[~temp_equity_curve.index.duplicated(keep='last')].sort_index()


    stats = {}
    if not trade_history_df.empty:
        st.write("Calcul stats & graphiques...")
        try:
            stats = calculate_statistics(trade_history_df, equity_curve_s, initial_equity, equity_history)
            if not equity_curve_s.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(equity_curve_s.index, equity_curve_s.values, label='Equity Curve', marker='.', linestyle='-', color='cyan')

                # --- CORRECTION appliquée ici ---
                ax.set_title('Backtest Equity Progression')
                ax.set_xlabel('Time')
                ax.set_ylabel('Equity ($)')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                # --- Fin Correction ---

                ax.set_facecolor('#111111'); fig.set_facecolor('#000000')
                ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                ax.title.set_color('white'); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
                if ax.get_legend(): ax.get_legend().get_texts()[0].set_color("white")
            st.write("Calculs terminés.")
        except Exception as final_calc_err:
            st.error(f"Erreur lors finalisation stats/graph: {final_calc_err}"); st.error(traceback.format_exc());
            equity_curve_s = pd.Series(dtype=float);
            stats = calculate_statistics(trade_history_df, pd.Series(dtype=float), initial_equity, equity_history);
            fig = None
    else:
        stats=calculate_statistics(pd.DataFrame(), pd.Series(dtype=float), initial_equity, [initial_equity]);
        st.write("Aucun trade exécuté ou fermé.")

    return trade_history_df, equity_curve_s, stats, fig, equity_history


# ==============================================================
# Fonction Plot Single Trade (Identique v3.2)
# ==============================================================
def plot_single_trade(file_path, trade_info, params):
    # (Code identique à la version précédente)
    try:
        entry_time=trade_info['entry_time']; exit_time=trade_info['exit_time']; entry_price=trade_info['entry_price']; exit_price=trade_info['exit_price']; stop_loss=trade_info['stop_loss']; take_profit=trade_info['take_profit']; trade_type=trade_info['entry_type']

        time_diff = exit_time - entry_time if pd.notna(exit_time) and pd.notna(entry_time) else pd.Timedelta(hours=1)
        time_buffer = max(pd.Timedelta(minutes=120), time_diff * 1.5)
        plot_start_time = entry_time - time_buffer if pd.notna(entry_time) else None
        plot_end_time = exit_time + time_buffer if pd.notna(exit_time) else (entry_time + time_buffer * 2 if pd.notna(entry_time) else None)

        if plot_start_time is None or plot_end_time is None:
             st.warning("Impossible de déterminer la fenêtre de temps du trade.")
             return None

        if not os.path.exists(file_path):
            st.error(f"Erreur plot: Fichier '{file_path}' introuvable.")
            return None

        df_full=pd.read_csv(file_path, parse_dates=['Open time'], index_col='Open time', encoding='utf-8', on_bad_lines='skip')
        if not pd.api.types.is_datetime64_any_dtype(df_full.index):
             df_full.index = pd.to_datetime(df_full.index, errors='coerce')
             df_full = df_full.dropna(subset=[df_full.index.name])

        df_full = df_full.sort_index().dropna(subset=['Open', 'High', 'Low', 'Close'])
        plot_df = df_full.loc[plot_start_time:plot_end_time].copy()

        if plot_df.empty: st.warning("Aucune donnée OHLC pour la fenêtre de temps du trade."); return None

        for col in ['Open','High','Low','Close']: plot_df[col]=pd.to_numeric(plot_df[col], errors='coerce')
        plot_df.dropna(subset=['Open','High','Low','Close'], inplace=True)
        if plot_df.empty: st.warning("Données OHLC invalides après conversion pour ce trade."); return None

        pine_ema1_p = params.get('pine_ema1_period'); pine_ema2_p = params.get('pine_ema2_period')
        if pine_ema1_p is not None and pine_ema1_p > 0 and pine_ema1_p < len(plot_df): plot_df['pine_ema1']=ta.trend.ema_indicator(plot_df['Close'],window=pine_ema1_p)
        if pine_ema2_p is not None and pine_ema2_p > 0 and pine_ema2_p < len(plot_df): plot_df['pine_ema2']=ta.trend.ema_indicator(plot_df['Close'],window=pine_ema2_p)

        rsi_p = params.get('rsi_length')
        if rsi_p is not None and rsi_p > 0 and rsi_p < len(plot_df): plot_df['rsi']=ta.momentum.rsi(plot_df['Close'],window=rsi_p)

        adx_p = params.get('adx_period')
        adx_threshold = params.get('adx_threshold')
        if adx_p is not None and adx_p > 0 and adx_p*2 < len(plot_df):
            if not plot_df[['High', 'Low', 'Close']].isnull().any().any():
                try: import pandas_ta as pta; adx_df = plot_df.ta.adx(length=adx_p); plot_df['adx'] = adx_df[f'ADX_{adx_p}']; plot_df['di_pos'] = adx_df[f'DMP_{adx_p}']; plot_df['di_neg'] = adx_df[f'DMN_{adx_p}']
                except ImportError: plot_df['adx']=ta.trend.adx(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p); plot_df['di_pos']=ta.trend.adx_pos(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p); plot_df['di_neg']=ta.trend.adx_neg(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p)
            else: plot_df['adx']=pd.NA; plot_df['di_pos']=pd.NA; plot_df['di_neg']=pd.NA

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.65, 0.175, 0.175], subplot_titles=("Prix & Indicateurs", "RSI", "ADX / DI"))

        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='OHLC'), row=1, col=1)

        if 'pine_ema1' in plot_df.columns and not plot_df['pine_ema1'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['pine_ema1'],mode='lines',name=f'EMA({pine_ema1_p})',line=dict(color='red',width=1.5)),row=1,col=1)
        if 'pine_ema2' in plot_df.columns and not plot_df['pine_ema2'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['pine_ema2'],mode='lines',name=f'EMA({pine_ema2_p})',line=dict(color='blue',width=1.5)),row=1,col=1)

        lec, ltp, lsl, lex = 'grey', 'lime', 'red', 'fuchsia'; pos_right="bottom right"; pos_left="top right";
        fig.add_hline(y=entry_price, line_dash="dash", line_color=lec, annotation_text=f"Entrée {entry_price:.5f}", annotation_position=pos_right, row=1, col=1);
        fig.add_hline(y=take_profit, line_dash="dot", line_color=ltp, annotation_text=f"TP {take_profit:.5f}", annotation_position=pos_right, row=1, col=1);
        fig.add_hline(y=stop_loss, line_dash="dot", line_color=lsl, annotation_text=f"SL {stop_loss:.5f}", annotation_position=pos_left, row=1, col=1);
        if not pd.isna(exit_price): fig.add_hline(y=exit_price, line_dash="dashdot", line_color=lex, annotation_text=f"Sortie {exit_price:.5f}", annotation_position=pos_left, row=1, col=1)

        lemc="rgba(100,100,255,0.5)";
        if pd.notna(entry_time): fig.add_vline(x=entry_time, line_width=1, line_dash="dash", line_color=lemc);
        if pd.notna(exit_time): fig.add_vline(x=exit_time, line_width=1, line_dash="dash", line_color=lemc)

        ms = 'triangle-up' if trade_type=='long' else 'triangle-down'; mc = 'lime' if trade_type=='long' else 'red'; mec='fuchsia';
        if pd.notna(entry_time) and pd.notna(entry_price): fig.add_trace(go.Scatter(x=[entry_time], y=[entry_price], mode='markers', name='Entrée Pt', marker=dict(symbol=ms, color=mc, size=12, line=dict(width=1,color='white'))), row=1, col=1);
        if pd.notna(exit_price) and pd.notna(exit_time): fig.add_trace(go.Scatter(x=[exit_time], y=[exit_price], mode='markers', name='Sortie Pt', marker=dict(symbol='x', color=mec, size=10, line=dict(width=1,color='white'))), row=1, col=1)

        if 'rsi' in plot_df.columns and not plot_df['rsi'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['rsi'], mode='lines', name='RSI', line=dict(color='rgb(180,180,255)', width=1)), row=2, col=1); rsi_ob = params.get('rsi_overbought'); rsi_os = params.get('rsi_oversold');
        if rsi_ob is not None: fig.add_hline(y=rsi_ob, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"OB({rsi_ob})", annotation_position="bottom right");
        if rsi_os is not None: fig.add_hline(y=rsi_os, line_dash="dash", line_color="lime", row=2, col=1, annotation_text=f"OS({rsi_os})", annotation_position="bottom right")

        if 'adx' in plot_df.columns and not plot_df['adx'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['adx'], mode='lines', name='ADX', line=dict(color='white', width=1.5)), row=3, col=1);
        if 'di_pos' in plot_df.columns and not plot_df['di_pos'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['di_pos'], mode='lines', name='+DI', line=dict(color='green', width=1)), row=3, col=1);
        if 'di_neg' in plot_df.columns and not plot_df['di_neg'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['di_neg'], mode='lines', name='-DI', line=dict(color='red', width=1)), row=3, col=1);
        if adx_threshold is not None and adx_threshold > 0: fig.add_hline(y=adx_threshold, line_dash="dot", line_color="aqua", name=f'Seuil ADX ({adx_threshold:.1f})', row=3, col=1)

        fig.update_layout(title=f"Visualisation Trade #{trade_info.name} ({trade_type.upper()})", xaxis_rangeslider_visible=False, height=800, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1), template="plotly_dark")
        fig.update_yaxes(title_text="Prix", row=1, col=1, gridcolor='rgba(180,180,180,0.3)');
        fig.update_yaxes(title_text="RSI", range=[0,100], row=2, col=1, gridcolor='rgba(180,180,180,0.3)');
        fig.update_yaxes(title_text="ADX/DI", row=3, col=1, gridcolor='rgba(180,180,180,0.3)');
        fig.update_xaxes(gridcolor='rgba(180,180,180,0.3)')
        return fig
    except FileNotFoundError:
        st.error(f"Erreur plot: Fichier '{file_path}' introuvable.")
        return None
    except Exception as e: st.error(f"Erreur lors de la génération du graphique de trade: {e}"); st.error(traceback.format_exc()); return None


# ==============================================================
# Fonction pour Graphique Explicatif (Corrigée v3.4.1)
# ==============================================================
def plot_strategy_explanation(ema1_p, ema2_p, rsi_p, rsi_os, rsi_ob, sl_pct, tp_rr):
    """Génère un graphique Plotly illustrant les règles de la stratégie Main."""
    try:
        n_points = 150
        index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_points, freq='h')) # Utilise 'h'

        base_price = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
        price_oscillation = 5 * np.sin(np.linspace(0, 10 * np.pi, n_points))
        close_np = base_price + price_oscillation
        high_np = close_np + np.abs(np.random.randn(n_points)) * 0.5
        low_np = close_np - np.abs(np.random.randn(n_points)) * 0.5

        # Convertir en Séries Pandas
        close = pd.Series(close_np, index=index, name='Close')
        high = pd.Series(high_np, index=index, name='High')
        low = pd.Series(low_np, index=index, name='Low')
        open_ = (close.shift(1) + low.shift(1)) / 2
        open_.iloc[0] = close.iloc[0] * 0.99
        open_.name = 'Open'

        df_expl = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close})
        df_expl.dropna(inplace=True)

        n_points = len(df_expl)
        if n_points < max(ema1_p, ema2_p, rsi_p, 2):
            st.warning("Données fictives insuffisantes pour calculer les indicateurs de l'explication.")
            return None

        try:
            if ema1_p < n_points: df_expl['ema1'] = ta.trend.ema_indicator(df_expl['Close'], window=ema1_p)
            if ema2_p < n_points: df_expl['ema2'] = ta.trend.ema_indicator(df_expl['Close'], window=ema2_p)
            if rsi_p < n_points: df_expl['rsi'] = ta.momentum.rsi(df_expl['Close'], window=rsi_p)
            df_expl.dropna(inplace=True)
            if df_expl.empty:
                st.warning("Données fictives insuffisantes après calcul des indicateurs.")
                return None
        except Exception as e_ind:
            st.warning(f"Erreur calcul indicateurs fictifs: {e_ind}")
            return None

        long_entry_idx = None; short_entry_idx = None
        if 'ema1' in df_expl and 'ema2' in df_expl and 'rsi' in df_expl:
            long_cond = (df_expl['Close'] > df_expl['ema1']) & (df_expl['ema1'] > df_expl['ema2']) & (df_expl['rsi'] < rsi_os)
            short_cond = (df_expl['Close'] < df_expl['ema1']) & (df_expl['ema1'] < df_expl['ema2']) & (df_expl['rsi'] > rsi_ob)
            long_entry_idx = df_expl[long_cond].first_valid_index()
            short_entry_idx = df_expl[short_cond].first_valid_index()

        fig_expl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                subplot_titles=("Exemple Conditions Entrée / SL / TP", "RSI"))

        fig_expl.add_trace(go.Candlestick(x=df_expl.index, open=df_expl['Open'], high=df_expl['High'], low=df_expl['Low'], close=df_expl['Close'], name='Prix'), row=1, col=1)

        if 'ema1' in df_expl: fig_expl.add_trace(go.Scatter(x=df_expl.index, y=df_expl['ema1'], mode='lines', name=f'EMA({ema1_p})', line=dict(color='red', width=1.5)), row=1, col=1)
        if 'ema2' in df_expl: fig_expl.add_trace(go.Scatter(x=df_expl.index, y=df_expl['ema2'], mode='lines', name=f'EMA({ema2_p})', line=dict(color='blue', width=1.5)), row=1, col=1)

        if 'rsi' in df_expl: fig_expl.add_trace(go.Scatter(x=df_expl.index, y=df_expl['rsi'], mode='lines', name='RSI', line=dict(color='rgb(180,180,255)', width=1)), row=2, col=1)
        fig_expl.add_hline(y=rsi_ob, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"Overbought ({rsi_ob})", annotation_position="top right")
        fig_expl.add_hline(y=rsi_os, line_dash="dash", line_color="lime", row=2, col=1, annotation_text=f"Oversold ({rsi_os})", annotation_position="bottom right")

        if long_entry_idx:
            entry_price_l = df_expl.loc[long_entry_idx, 'Close']
            sl_price_l = entry_price_l * (1 - sl_pct)
            risk_unit_l = entry_price_l - sl_price_l
            tp_price_l = entry_price_l + (risk_unit_l * tp_rr)
            fig_expl.add_vline(x=long_entry_idx, line_width=1, line_dash="dash", line_color="lime", row=1, col=1)
            fig_expl.add_trace(go.Scatter(x=[long_entry_idx], y=[entry_price_l], mode='markers', name='Exemple Long', marker=dict(symbol='triangle-up', color='lime', size=12)), row=1, col=1)
            fig_expl.add_hline(y=sl_price_l, line_dash="dot", line_color="orange", row=1, col=1, annotation_text=f"SL Long ({sl_price_l:.2f})", annotation_position="bottom left")
            fig_expl.add_hline(y=tp_price_l, line_dash="dot", line_color="fuchsia", row=1, col=1, annotation_text=f"TP Long ({tp_price_l:.2f})", annotation_position="top left")
            fig_expl.add_annotation(x=long_entry_idx, y=df_expl['High'].max()*1.01, text=f"Cond. Long: P > EMA({ema1_p}) > EMA({ema2_p}) & RSI < {rsi_os}", showarrow=False, bgcolor="rgba(0,100,0,0.6)", row=1, col=1, xanchor="left", yanchor="top")

        if short_entry_idx:
            entry_price_s = df_expl.loc[short_entry_idx, 'Close']
            sl_price_s = entry_price_s * (1 + sl_pct)
            risk_unit_s = sl_price_s - entry_price_s
            tp_price_s = entry_price_s - (risk_unit_s * tp_rr)
            fig_expl.add_vline(x=short_entry_idx, line_width=1, line_dash="dash", line_color="red", row=1, col=1)
            fig_expl.add_trace(go.Scatter(x=[short_entry_idx], y=[entry_price_s], mode='markers', name='Exemple Short', marker=dict(symbol='triangle-down', color='red', size=12)), row=1, col=1)
            fig_expl.add_hline(y=sl_price_s, line_dash="dot", line_color="orange", row=1, col=1, annotation_text=f"SL Short ({sl_price_s:.2f})", annotation_position="top right")
            fig_expl.add_hline(y=tp_price_s, line_dash="dot", line_color="fuchsia", row=1, col=1, annotation_text=f"TP Short ({tp_price_s:.2f})", annotation_position="bottom right")
            fig_expl.add_annotation(x=short_entry_idx, y=df_expl['Low'].min()*0.99, text=f"Cond. Short: P < EMA({ema1_p}) < EMA({ema2_p}) & RSI > {rsi_ob}", showarrow=False, bgcolor="rgba(100,0,0,0.6)", row=1, col=1, xanchor="right", yanchor="bottom")

        fig_expl.update_layout(title="Illustration des Règles de la Stratégie 'Main'", height=600, template="plotly_dark", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_expl.update_yaxes(title_text="Prix", row=1, col=1, gridcolor='rgba(180,180,180,0.3)')
        fig_expl.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, gridcolor='rgba(180,180,180,0.3)')
        fig_expl.update_xaxes(gridcolor='rgba(180,180,180,0.3)')
        fig_expl.update_layout(xaxis_rangeslider_visible=False)

        return fig_expl
    except Exception as e_expl:
        st.error(f"Erreur lors de la génération du graphique explicatif: {e_expl}")
        st.error(traceback.format_exc())
        return None


# ==============================================================
# Fonction pour Calcul Rentabilité Périodique (Identique v3.3)
# ==============================================================
def calculate_periodical_returns(trade_history_df):
    # (Code identique à la version précédente)
    if not isinstance(trade_history_df, pd.DataFrame) or trade_history_df.empty:
        return {'yearly': pd.DataFrame(), 'monthly': pd.DataFrame(), 'daily': pd.DataFrame()}

    df = trade_history_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['exit_time']):
        df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        df.dropna(subset=['exit_time'], inplace=True)

    if df.empty:
         return {'yearly': pd.DataFrame(), 'monthly': pd.DataFrame(), 'daily': pd.DataFrame()}

    df['Year'] = df['exit_time'].dt.year
    df['YearMonth'] = df['exit_time'].dt.to_period('M')
    df['Date'] = df['exit_time'].dt.date

    def win_rate(x):
        wins = (x > 0).sum()
        total = len(x)
        return (wins / total) * 100 if total > 0 else 0

    agg_funcs = {
        'profit': ['sum', 'mean', 'count', win_rate]
    }

    def format_results(df_agg):
        if df_agg.empty:
            return df_agg
        df_agg.columns = ['Total Profit', 'Avg Profit/Trade', 'Nb Trades', 'Win Rate %']
        df_agg['Win Rate %'] = df_agg['Win Rate %'].round(2)
        df_agg['Total Profit'] = df_agg['Total Profit'].round(2)
        df_agg['Avg Profit/Trade'] = df_agg['Avg Profit/Trade'].round(2)
        return df_agg

    try:
        yearly_returns = df.groupby('Year').agg(agg_funcs)
        yearly_returns = format_results(yearly_returns)
    except Exception as e:
        st.warning(f"Erreur calcul rentabilité annuelle: {e}")
        yearly_returns = pd.DataFrame()

    try:
        monthly_returns = df.groupby('YearMonth').agg(agg_funcs)
        monthly_returns = format_results(monthly_returns)
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
    except Exception as e:
        st.warning(f"Erreur calcul rentabilité mensuelle: {e}")
        monthly_returns = pd.DataFrame()

    try:
        daily_returns = df.groupby('Date').agg(agg_funcs)
        daily_returns = format_results(daily_returns)
    except Exception as e:
        st.warning(f"Erreur calcul rentabilité journalière: {e}")
        daily_returns = pd.DataFrame()

    return {'yearly': yearly_returns, 'monthly': monthly_returns, 'daily': daily_returns}


# ==============================================================
# --- Interface Utilisateur Streamlit (Identique v3.4) ---
# ==============================================================
# --- Barre Latérale (Sidebar) ---
st.sidebar.header("Paramètres Backtest 'Main'")

LOCAL_DATA_FILE = "BTC-USD_data_1min.csv"
if not os.path.exists(LOCAL_DATA_FILE):
    st.sidebar.error(f"Fichier '{LOCAL_DATA_FILE}' introuvable !")
st.sidebar.caption(f"Données: {LOCAL_DATA_FILE} (local)")

initial_equity = st.sidebar.number_input("Capital Initial ($)", min_value=1.0, value=5000.0, step=100.0, format="%.2f", key="init_eq_main")

st.sidebar.subheader("Période de Backtest")
default_start_date = datetime.date(2017, 1, 1)
default_end_date = datetime.date.today()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date_input = st.date_input("Date de Début", value=default_start_date, key="start_date")
with col_date2:
    end_date_input = st.date_input("Date de Fin", value=default_end_date, key="end_date")

dates_valid = True
if start_date_input > end_date_input:
    st.sidebar.error("La date de début doit être antérieure ou égale à la date de fin.")
    dates_valid = False

strategy_type_arg = 'pinescript_main'
st.sidebar.subheader("Indicateurs Stratégie 'Main'")
with st.sidebar.container(border=True):
    col_pine1, col_pine2 = st.columns(2)
    with col_pine1: pine_ema1_val = st.number_input("EMA 1 (Rapide)", min_value=2, value=1000, step=1, format="%d", key="pine_ema1")
    with col_pine2: pine_ema2_val = st.number_input("EMA 2 (Lente)", min_value=2, value=5000, step=1, format="%d", key="pine_ema2")
    rsi_len_param = st.number_input("Période RSI",min_value=2,max_value=100,value=14,step=1,format="%d",key="rsi_len_pine")
    col_rsi1, col_rsi2 = st.columns(2)
    with col_rsi1: rsi_os_val=st.number_input("RSI Oversold",min_value=1,max_value=50,value=30,step=1,format="%d", key="rsi_os_pine")
    with col_rsi2: rsi_ob_val=st.number_input("RSI Overbought",min_value=50,max_value=99,value=70,step=1,format="%d", key="rsi_ob_pine")

params_indic_valid = True
if pine_ema2_val <= pine_ema1_val: params_indic_valid = False; st.sidebar.error("EMA 2 <= EMA 1")
if rsi_ob_val <= rsi_os_val: params_indic_valid = False; st.sidebar.error("RSI OB <= RSI OS")

st.sidebar.subheader("Gestion du Risque et Sortie")
with st.sidebar.container(border=True):
    st.caption("SL & TP")
    pine_sl_pct_input = st.number_input("Stop Loss (%)", min_value=0.01, max_value=10.0, value=0.2, step=0.01, format="%.2f", key="pine_sl_pct")
    pine_tp_mult_input = st.number_input("Ratio Risque/Rendement (RR)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f", key="pine_rr")

    st.caption("Dimensionnement (Sizing)")
    sizing_mode_input = st.radio("Méthode Sizing", ["Risque %", "Lot Fixe"], index=0, key='sizing_mode_pine')
    sizing_type_arg = 'risk_pct' if sizing_mode_input == 'Risque %' else 'fixed_lot'

    pine_risk_pct_val = 0.005
    pine_fixed_lot_val = 0.01

    if sizing_type_arg == 'risk_pct':
        pine_risk_pct_input = st.number_input("Risque par Trade (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f", key="pine_risk_pct")
        pine_risk_pct_val = pine_risk_pct_input / 100.0
    else:
        pine_fixed_lot_input = st.number_input("Taille Lot Fixe", min_value=0.0001, value=0.01, step=0.001, format="%.4f", key="pine_fixed_lot")
        pine_fixed_lot_val = pine_fixed_lot_input

st.sidebar.subheader("Filtres Additionnels et Concurrence")
use_adx_filter_input = st.sidebar.toggle("Activer Filtre ADX", value=False, key="toggle_adx_filter")
adx_period_input = 14
adx_threshold_input = 25.0
calc_adx_for_load = use_adx_filter_input
if use_adx_filter_input:
    adx_period_input = st.sidebar.number_input("Période ADX (pour filtre)", min_value=2, max_value=100, value=14, step=1, format="%d", key="adx_p_filt")
    adx_threshold_input = st.sidebar.number_input("Seuil ADX Min. pour Trader", min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.1f", key="adx_thresh_filt")

use_min_profit_filter_input = st.sidebar.toggle( "Activer Filtre Min Profit Points", value=False, key="toggle_min_profit")
min_profit_points_input = 0.0
if use_min_profit_filter_input: min_profit_points_input = st.sidebar.number_input( "Seuil Min Profit Points (en points de prix)", min_value=0.0, value=50.0, step=1.0, format="%.5f", key="min_profit_val", help="Ex: Pour BTC/USD, 50 signifie 50$ de différence minimale entre entrée et TP.")
one_trade_at_a_time_input = st.sidebar.checkbox("Limiter à un seul trade ouvert", value=True)

# --- Bouton de Lancement ---
run_button_disabled = (not os.path.exists(LOCAL_DATA_FILE)) or (not params_indic_valid) or (not dates_valid)
run_button = st.sidebar.button("🚀 Lancer le Backtest", disabled=run_button_disabled, use_container_width=True)
st.sidebar.markdown("---"); st.sidebar.info("Backtester v3.4.1")

# --- Zone d'Affichage Principale ---
st.header("Résultats du Backtest")

# Afficher le graphique explicatif
with st.expander("🔍 Explication Visuelle de la Stratégie", expanded=False):
    # Utilise les valeurs actuelles de la sidebar pour l'explication
    fig_expl = plot_strategy_explanation(
        ema1_p=pine_ema1_val, ema2_p=pine_ema2_val,
        rsi_p=rsi_len_param, rsi_os=rsi_os_val, rsi_ob=rsi_ob_val,
        sl_pct=pine_sl_pct_input / 100.0, tp_rr=pine_tp_mult_input
    )
    if fig_expl:
        st.plotly_chart(fig_expl, use_container_width=True)
    else:
        st.warning("Impossible de générer le graphique explicatif.")

# Initialisation de l'état de session
if 'results_calculated' not in st.session_state: st.session_state.update({'results_calculated':False, 'trade_history':pd.DataFrame(), 'equity_curve':pd.Series(dtype=float), 'statistics':{}, 'equity_fig':None, 'backtest_params':{}, 'raw_equity_history': [], 'periodical_returns': {}})

if run_button:
    st.session_state.update({'results_calculated':False, 'trade_history':pd.DataFrame(), 'equity_curve':pd.Series(dtype=float), 'statistics':{}, 'equity_fig':None, 'raw_equity_history': [], 'periodical_returns': {}})

    strat_display_name = f"PineScript Main [EMA({pine_ema1_val}/{pine_ema2_val}), RSI({rsi_len_param},{rsi_os_val}/{rsi_ob_val})]"
    filter_info = []
    if use_adx_filter_input: filter_info.append(f"Filtre ADX({adx_period_input})>{adx_threshold_input:.1f}")
    if use_min_profit_filter_input: filter_info.append(f"Filtre MinPtsTP>{min_profit_points_input:.5f}")
    filter_str = ", ".join(filter_info) if filter_info else "Aucun"
    sizing_info = f"{pine_risk_pct_input:.2f}% Risk" if sizing_type_arg == 'risk_pct' else f"{pine_fixed_lot_val:.4f} Lot Fixe"
    sl_info = f"{pine_sl_pct_input:.2f}% Px"
    rr_info = f"{pine_tp_mult_input:.1f}"
    concurrency_mode = "Unique" if one_trade_at_a_time_input else "Multiple";
    info_str = f"Lancement: {strat_display_name} | Sizing:{sizing_info} | SL:{sl_info} | RR:{rr_info} | Filtres:[{filter_str}] | Conc:{concurrency_mode}"; st.info(info_str); progress_placeholder_area = st.empty()

    current_params = {
        "strategy_type": strategy_type_arg, "pine_ema1_period": pine_ema1_val, "pine_ema2_period": pine_ema2_val,
        "rsi_length": rsi_len_param, "rsi_oversold": rsi_os_val, "rsi_overbought": rsi_ob_val,
        "adx_period": adx_period_input if calc_adx_for_load else None, "adx_threshold": adx_threshold_input if use_adx_filter_input else None,
        "atr_period": None, "atr_threshold": None, "ema_short_period": None, "ema_long_period": None,
        "reversal_ma_period": None, "div_lookback_period": None
    }; st.session_state.backtest_params = current_params

    st.write(f"Préparation données depuis '{LOCAL_DATA_FILE}' (via cache)...");
    df_full_with_indicators = load_data_and_indicators(file_path=LOCAL_DATA_FILE,
                                               pine_ema1_p=pine_ema1_val,
                                               pine_ema2_p=pine_ema2_val,
                                               rsi_p=rsi_len_param,
                                               calc_adx=calc_adx_for_load,
                                               adx_p=adx_period_input)

    if not df_full_with_indicators.empty:
        st.write("Données chargées. Filtrage par dates...")
        start_datetime = pd.to_datetime(start_date_input)
        # Assurer que la date de fin inclut toute la journée sélectionnée
        end_datetime = pd.to_datetime(end_date_input) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        try:
            df_filtered = df_full_with_indicators.loc[start_datetime:end_datetime].copy()

            if df_filtered.empty:
                st.warning(f"Aucune donnée disponible pour la période sélectionnée ({start_date_input.strftime('%Y-%m-%d')} au {end_date_input.strftime('%Y-%m-%d')}).")
                st.session_state.results_calculated = False
            else:
                 st.write(f"Données filtrées prêtes pour backtest ({len(df_filtered)} lignes). Période réelle testée: {df_filtered.index.min()} -> {df_filtered.index.max()}")
                 with st.spinner("Backtest en cours..."):
                     th, ec, stats, efig, raw_eq_hist = backtest_strategy(
                         df_processed=df_filtered, initial_equity=initial_equity,
                         pine_ema1_period=pine_ema1_val, pine_ema2_period=pine_ema2_val,
                         rsi_length=rsi_len_param, rsi_oversold=rsi_os_val, rsi_overbought=rsi_ob_val,
                         pine_sl_percentage=pine_sl_pct_input / 100.0,
                         sizing_type=sizing_type_arg, pine_risk_percentage=pine_risk_pct_val, pine_fixed_lot_size=pine_fixed_lot_val,
                         pine_tp_multiplier=pine_tp_mult_input, use_adx_filter=use_adx_filter_input, adx_threshold=adx_threshold_input,
                         one_trade_at_a_time=one_trade_at_a_time_input, use_min_profit_points_filter=use_min_profit_filter_input,
                         min_profit_points_threshold=min_profit_points_input, progress_placeholder=progress_placeholder_area
                     )
                 progress_placeholder_area.empty();
                 periodical_returns_data = calculate_periodical_returns(th)
                 st.session_state.trade_history=th; st.session_state.equity_curve=ec; st.session_state.statistics=stats; st.session_state.equity_fig=efig; st.session_state.raw_equity_history = raw_eq_hist; st.session_state.results_calculated=True; st.session_state.periodical_returns = periodical_returns_data
                 st.success("Backtest terminé !")

        except Exception as filter_err:
            st.error(f"Erreur lors du filtrage des données par date: {filter_err}")
            st.session_state.results_calculated = False
    else: st.error("Chargement/préparation données échoué."); st.session_state.results_calculated = False

# Affichage des résultats si calculés
if st.session_state.results_calculated:
    stats=st.session_state.statistics; equity_fig=st.session_state.equity_fig; trade_history=st.session_state.trade_history; equity_curve=st.session_state.equity_curve; raw_equity_history = st.session_state.raw_equity_history; periodical_returns = st.session_state.periodical_returns

    if stats and isinstance(stats, dict) and 'Final Equity' in stats:
        st.subheader("Période de Trading (Résultats)");
        first_trade_dt = pd.to_datetime(stats.get('First Trade Date'))
        last_trade_dt = pd.to_datetime(stats.get('Last Trade Date'))
        date_format='%Y-%m-%d %H:%M:%S';
        date_col1, date_col2 = st.columns(2);
        with date_col1: st.markdown(f"**Premier Trade:**"); st.write(first_trade_dt.strftime(date_format) if pd.notna(first_trade_dt) else "N/A");
        with date_col2: st.markdown(f"**Dernier Trade:**"); st.write(last_trade_dt.strftime(date_format) if pd.notna(last_trade_dt) else "N/A");
        st.caption(f"Période de backtest sélectionnée : {start_date_input.strftime('%Y-%m-%d')} à {end_date_input.strftime('%Y-%m-%d')}")
        st.divider()

        st.subheader("Statistiques Clés"); col1,col2,col3=st.columns(3); col1.metric("Profit ($)",f"{stats.get('Total Profit',0):,.2f}"); col2.metric("Profit (%)",f"{stats.get('Profit (%)',0):.2f}%"); pf_val = stats.get('Profit Factor', 0); col3.metric("PF",f"{pf_val:.2f}" if pf_val != float('inf') else "Inf")
        col4,col5,col6=st.columns(3); col4.metric("Trades",f"{stats.get('Number of Trades',0):,}"); col5.metric("Win (%)",f"{stats.get('Winning Trades (%)',0):.2f}%"); col6.metric("Max DD (%)",f"{stats.get('Max Drawdown (%)',0):.2f}%")
        col7,col8,col9=st.columns(3); col7.metric("Max Loss Str",f"{stats.get('Max Consecutive Losing Trades',0)}"); col8.metric("Avg Loss Str",f"{stats.get('Average Consecutive Losing Trades',0):.1f}"); col9.metric("Final Eq ($)",f"{stats.get('Final Equity', initial_equity):,.2f}"); st.divider()

        st.subheader("Courbe d'Équité");
        if equity_fig: st.pyplot(equity_fig); plt.close(equity_fig)
        elif stats.get('Number of Trades', 0) > 0 : st.warning("Graphique équité non généré.")
        else: st.info("Aucun trade exécuté, pas de courbe d'équité.")

        st.divider()
        st.subheader("Rentabilité Périodique")
        if periodical_returns and isinstance(periodical_returns, dict):
             if not periodical_returns.get('yearly', pd.DataFrame()).empty:
                 with st.expander("🗓️ Rentabilité Annuelle", expanded=False):
                     df_yearly = periodical_returns['yearly']
                     st.dataframe(df_yearly.style.format("{:.2f}", subset=['Total Profit', 'Avg Profit/Trade', 'Win Rate %']))
                     if 'Total Profit' in df_yearly.columns:
                         st.bar_chart(df_yearly['Total Profit'])
             else: st.info("Aucune donnée annuelle.")

             if not periodical_returns.get('monthly', pd.DataFrame()).empty:
                 with st.expander("🗓️ Rentabilité Mensuelle", expanded=False):
                      df_monthly = periodical_returns['monthly']
                      st.dataframe(df_monthly.style.format("{:.2f}", subset=['Total Profit', 'Avg Profit/Trade', 'Win Rate %']))
                      if 'Total Profit' in df_monthly.columns:
                            try:
                                chart_data = df_monthly.reset_index().rename(columns={'index': 'YearMonth'})
                                st.bar_chart(chart_data, x='YearMonth', y='Total Profit')
                            except Exception as e:
                                st.warning(f"Impossible d'afficher le graphique mensuel: {e}")
             else: st.info("Aucune donnée mensuelle.")

             if not periodical_returns.get('daily', pd.DataFrame()).empty:
                 with st.expander("🗓️ Rentabilité Journalière", expanded=False):
                     df_daily = periodical_returns['daily']
                     st.dataframe(df_daily.style.format("{:.2f}", subset=['Total Profit', 'Avg Profit/Trade', 'Win Rate %']))
             else: st.info("Aucune donnée journalière.")
        else:
            st.info("Aucune donnée de rentabilité périodique à afficher (aucun trade ou erreur de calcul).")

        st.divider()
        st.subheader("Historique des Trades")
        if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
            display_df = pd.concat([trade_history.head(),trade_history.tail()]) if len(trade_history) > 10 else trade_history
            try:
                 display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                 display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
            except: pass
            st.dataframe(display_df.style.format({"entry_price":"{:.5f}", "exit_price":"{:.5f}", "stop_loss":"{:.5f}", "take_profit":"{:.5f}", "profit":"{:,.2f}", "size":"{:.4f}"}))
            csv_data = trade_history.to_csv(index=False).encode('utf-8'); st.download_button(label="📥 Télécharger historique (CSV)", data=csv_data, file_name='trade_history_main.csv', mime='text/csv')
        elif stats.get('Number of Trades',-1)==0: st.info("Aucun trade exécuté pour la période sélectionnée.")
        else: st.warning("Historique des trades non disponible ou vide.")

        st.divider()
        st.subheader("Visualisation d'un Trade Spécifique")
        if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
            max_trade_index = len(trade_history) - 1; trade_indices = list(range(len(trade_history))); default_idx = 0 if max_trade_index >= 0 else None
            if default_idx is not None:
                selected_trade_idx_ui = st.selectbox(f"Choisir index trade (0 à {max_trade_index})", options=trade_indices, index=default_idx, key='trade_selector_idx_main')
                if selected_trade_idx_ui is not None and st.button("Afficher Graphique Trade", key='show_trade_btn_main'):
                    with st.spinner("Génération graphique..."):
                        trade_details = trade_history.iloc[selected_trade_idx_ui];
                        backtest_params_for_plot = st.session_state.backtest_params;
                        single_trade_fig = plot_single_trade(LOCAL_DATA_FILE, trade_details, backtest_params_for_plot)
                        if single_trade_fig: st.plotly_chart(single_trade_fig, use_container_width=True)
                        else: st.warning("Impossible d'afficher graphique trade.")
            else: st.info("Aucun trade disponible.")
        else: st.info("Aucun trade dans l'historique pour visualiser.")

    elif run_button and not (stats and isinstance(stats, dict) and 'Final Equity' in stats):
         st.warning("Le backtest s'est terminé mais aucune statistique valide n'a pu être calculée. Vérifiez la plage de dates ou les logs d'erreur.")
    elif not params_indic_valid:
         st.warning("Paramètres d'indicateurs invalides détectés.")
    elif not dates_valid:
         st.warning("Plage de dates invalide sélectionnée.")
    elif run_button and not os.path.exists(LOCAL_DATA_FILE):
        st.error(f"Le fichier de données '{LOCAL_DATA_FILE}' est introuvable. Le backtest n'a pas pu démarrer.")
    else:
        if not run_button and not st.session_state.results_calculated:
             st.info("Configurez les paramètres et lancez le backtest pour voir les résultats.")
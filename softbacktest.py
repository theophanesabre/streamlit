# -*- coding: utf-8 -*-
# gui_backtester_spread_choice.py

import streamlit as st
import pandas as pd
import ta
import matplotlib.pyplot as plt
import matplotlib
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use('Agg')

# ==============================================================
# Fonction de Calcul des Statistiques (INCHANGÉE)
# ==============================================================
def calculate_statistics(trade_history, equity_curve, initial_equity):
    """Identique à la version précédente"""
    stats = {}
    number_of_trades = len(trade_history)
    stats['Number of Trades'] = number_of_trades
    stats['First Trade Date'], stats['Last Trade Date'] = None, None
    stats['Total Profit'], stats['Final Equity'] = 0, initial_equity
    stats['Profit (%)'], stats['Winning Trades (%)'] = 0, 0
    stats['Max Drawdown (%)'], stats['Max Consecutive Losing Trades'] = 0, 0
    stats['Average Consecutive Losing Trades'], stats['Average Profit per Trade'] = 0, 0
    stats['Profit Factor'] = 0
    if number_of_trades == 0 or equity_curve.empty: return stats
    stats['First Trade Date'] = trade_history['entry_time'].iloc[0]
    stats['Last Trade Date'] = trade_history['exit_time'].iloc[-1]
    total_profit = trade_history['profit'].sum()
    stats['Total Profit'] = total_profit
    final_equity = initial_equity + total_profit
    stats['Final Equity'] = final_equity
    stats['Profit (%)'] = (total_profit / initial_equity) * 100 if initial_equity > 0 else 0
    winning_trades = trade_history[trade_history['profit'] > 0]
    losing_trades = trade_history[trade_history['profit'] <= 0]
    stats['Winning Trades (%)'] = len(winning_trades) / number_of_trades * 100
    temp_equity_curve = pd.concat([pd.Series([initial_equity], index=[equity_curve.index.min() - pd.Timedelta(seconds=1)]), equity_curve])
    max_drawdown, current_peak = 0, initial_equity
    for val in temp_equity_curve:
         current_peak = max(current_peak, val)
         drawdown = (current_peak - val) / current_peak * 100 if current_peak > 0 else 0
         max_drawdown = max(max_drawdown, drawdown)
    stats['Max Drawdown (%)'] = max_drawdown
    losing_streak, max_losing_streak, losing_streak_lengths = 0, 0, []
    for profit in trade_history['profit']:
        if profit <= 0: losing_streak += 1
        else:
            if losing_streak > 0: losing_streak_lengths.append(losing_streak)
            max_losing_streak = max(max_losing_streak, losing_streak)
            losing_streak = 0
    if losing_streak > 0: losing_streak_lengths.append(losing_streak)
    max_losing_streak = max(max_losing_streak, losing_streak)
    stats['Max Consecutive Losing Trades'] = max_losing_streak
    if losing_streak_lengths: average_losing_streak = sum(losing_streak_lengths) / len(losing_streak_lengths)
    else: average_losing_streak = 0
    stats['Average Consecutive Losing Trades'] = average_losing_streak
    stats['Average Profit per Trade'] = total_profit / number_of_trades
    gross_profit = winning_trades['profit'].sum()
    gross_loss = abs(losing_trades['profit'].sum())
    stats['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    return stats

# ==============================================================
# Fonction Principale de Backtesting (MODIFIÉE pour choix Spread)
# ==============================================================
def backtest_strategy(csv_filepath, initial_equity=5000, ema_short_period=1000,
                      ema_long_period=5000, rsi_length=14, rsi_oversold=30,
                      rsi_overbought=70, risk_percentage=0.005,
                      sl_type='percentage', stop_loss_percentage=0.002,
                      atr_period=14, atr_multiplier_sl=2.0,
                      atr_threshold=0.0,
                      # --- NOUVEAU: Paramètres Spread multiples ---
                      spread_type='fixed', # 'fixed' or 'percentage'
                      spread_cost=0.0,     # Valeur fixe en $
                      spread_percentage=0.0, # Valeur % en décimal (ex: 0.0005 pour 0.05%)
                      # --- FIN NOUVEAU ---
                      take_profit_multiplier=5,
                      progress_placeholder=None,
                      one_trade_at_a_time=True):
    """
    Effectue le backtest avec choix SL/Spread, filtre ATR, et gestion concurrence.
    """
    fig = None
    # --- Lecture CSV et préparation (INCHANGÉ) ---
    try:
        df = pd.read_csv(csv_filepath)
        try: df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception: df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
        if df['timestamp'].isnull().all(): st.error(f"Critique: Colonne 'Date' non convertible."); return pd.DataFrame(), pd.Series(dtype=float), {}, fig
        df.set_index('timestamp', inplace=True); df = df.sort_index()
    except FileNotFoundError: st.error(f"Erreur: Fichier '{csv_filepath}' introuvable."); return pd.DataFrame(), pd.Series(dtype=float), {}, fig
    except KeyError: st.error(f"Erreur: Colonne 'Date' manquante."); return pd.DataFrame(), pd.Series(dtype=float), {}, fig
    except Exception as e: st.error(f"Erreur lecture CSV: {e}"); return pd.DataFrame(), pd.Series(dtype=float), {}, fig

    # --- Nettoyage et Validation OHLC (INCHANGÉ) ---
    numeric_cols=['Open','High','Low','Close']; missing_cols=[c for c in numeric_cols if c not in df.columns]
    if missing_cols: st.error(f"Erreur: Colonnes OHLC manquantes: {', '.join(missing_cols)}"); return pd.DataFrame(), pd.Series(dtype=float), {}, fig
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    initial_rows=len(df); df.dropna(subset=numeric_cols, inplace=True); rows_dropped=initial_rows-len(df)
    if rows_dropped > 0: st.warning(f"{rows_dropped} lignes supprimées (OHLC invalides).")
    required_length = max(ema_long_period, rsi_length, atr_period) + 1
    if len(df) < required_length: st.error(f"Erreur: Pas assez de données ({len(df)}) (Min: {required_length})."); return pd.DataFrame(), pd.Series(dtype=float), {}, fig

    # --- Calcul Indicateurs (INCHANGÉ) ---
    try:
        df['ema_short']=ta.trend.ema_indicator(df['Close'],window=ema_short_period); df['ema_long']=ta.trend.ema_indicator(df['Close'],window=ema_long_period)
        df['rsi']=ta.momentum.rsi(df['Close'],window=rsi_length); df['atr']=ta.volatility.average_true_range(df['High'],df['Low'],df['Close'],window=atr_period)
        df.dropna(subset=['ema_long', 'rsi', 'atr'], inplace=True)
        if df.empty: st.error("Erreur: Aucune donnée après calcul indicateurs."); return pd.DataFrame(), pd.Series(dtype=float), {}, fig
    except Exception as e: st.error(f"Erreur calcul indicateurs: {e}"); return pd.DataFrame(), pd.Series(dtype=float), {}, fig

    # --- Initialisation Backtest (INCHANGÉ) ---
    closed_trades_history=[]; equity_history=[initial_equity]; equity=initial_equity
    open_positions=[]; trade_id_counter=0
    progress_placeholder_area = st.empty(); total_rows=len(df)

    # --- Boucle Principale ---
    for i, (index, row) in enumerate(df.iterrows()):
        # MAJ Placeholder (INCHANGÉ)
        if progress_placeholder and (i % 500 == 0 or i == total_rows - 1):
            prog=float(i+1)/total_rows; perc=min(int(prog*100),100); progress_placeholder.text(f"Progression: {perc}%")

        signal_price=row['Close']; current_high=row['High']; current_low=row['Low']
        ema_short=row['ema_short']; ema_long=row['ema_long']; rsi=row['rsi']; current_atr=row['atr']
        if pd.isna(signal_price) or signal_price<=0 or pd.isna(current_atr) or current_atr<=1e-9: continue

        # --- Logique de Sortie (INCHANGÉE) ---
        positions_to_remove=[]; equity_at_start_of_bar=equity
        for position in open_positions:
            exit_price=None; pos_id=position['id']; pos_entry_price=position['entry_price']; pos_type=position['type']
            pos_sl=position['stop_loss']; pos_tp=position['take_profit']; pos_size=position['size']; pos_entry_time=position['entry_time']
            if pos_type=='long':
                if current_low<=pos_sl: exit_price=pos_sl
                elif current_high>=pos_tp: exit_price=pos_tp
            elif pos_type=='short':
                if current_high>=pos_sl: exit_price=pos_sl
                elif current_low<=pos_tp: exit_price=pos_tp
            if exit_price is not None:
                if pos_type=='long': profit=(exit_price-pos_entry_price)*pos_size
                else: profit=(pos_entry_price-exit_price)*pos_size
                equity+=profit; equity=max(equity,0)
                closed_trades_history.append({'trade_id':pos_id,'entry_time':pos_entry_time,'entry_price':pos_entry_price,'entry_type':pos_type,'size':pos_size,'stop_loss':pos_sl,'take_profit':pos_tp,'exit_time':index,'exit_price':exit_price,'profit':profit})
                equity_history.append(equity); positions_to_remove.append(position)
        for closed_pos in positions_to_remove: open_positions.remove(closed_pos)

        # --- Logique d'Entrée ---
        if current_atr < atr_threshold: continue # Filtre ATR
        long_condition = signal_price > ema_short and ema_short > ema_long and rsi < rsi_oversold
        short_condition = signal_price < ema_short and ema_short < ema_long and rsi > rsi_overbought
        can_enter = True if not one_trade_at_a_time else (len(open_positions) == 0) # Condition Concurrence

        if can_enter and (long_condition or short_condition):
            risk_amount = equity*risk_percentage
            if equity <= 0: st.warning("Équité <= 0. Arrêt."); break
            actual_entry_price, stop_loss_price, risk_per_unit = None, None, 0

            # --- MODIFIÉ: Calcul Prix Entrée selon Type Spread ---
            if spread_type == 'fixed':
                if long_condition: actual_entry_price = signal_price + spread_cost
                else: actual_entry_price = signal_price - spread_cost
            elif spread_type == 'percentage':
                 # spread_percentage est déjà en décimal (ex: 0.0005 pour 0.05%)
                if long_condition: actual_entry_price = signal_price * (1 + spread_percentage)
                else: actual_entry_price = signal_price * (1 - spread_percentage)
            else:
                 st.error(f"Type de spread non reconnu: {spread_type}"); continue # Ignore le signal
            # --- FIN MODIFICATION ---

            # Calcul SL basé sur prix d'entrée AJUSTÉ (logique inchangée)
            if sl_type=='percentage':
                if long_condition: stop_loss_price=actual_entry_price*(1-stop_loss_percentage)
                else: stop_loss_price=actual_entry_price*(1+stop_loss_percentage)
            elif sl_type=='atr':
                if long_condition: stop_loss_price=actual_entry_price-(atr_multiplier_sl*current_atr)
                else: stop_loss_price=actual_entry_price+(atr_multiplier_sl*current_atr)
            else: continue # Type SL inconnu

            # Calcul Risque et Taille (inchangé en logique, mais utilise prix ajustés)
            risk_per_unit=abs(actual_entry_price-stop_loss_price)
            if risk_per_unit <= 1e-9: continue
            position_size = risk_amount / risk_per_unit

            if position_size > 0:
                # Calcul TP basé sur prix d'entrée AJUSTÉ
                if long_condition: take_profit_price=actual_entry_price+(risk_per_unit*take_profit_multiplier)
                else: take_profit_price=actual_entry_price-(risk_per_unit*take_profit_multiplier)

                # Enregistrement nouvelle position (utilise actual_entry_price)
                trade_id_counter+=1
                new_position = {'id':trade_id_counter,'entry_time':index,'entry_price':actual_entry_price,
                                'type':'long' if long_condition else 'short','size':position_size,
                                'stop_loss':stop_loss_price,'take_profit':take_profit_price}
                open_positions.append(new_position)

    # --- Finalisation et Retour (INCHANGÉ) ---
    trade_history_df=pd.DataFrame(closed_trades_history)
    if not trade_history_df.empty:
         trade_history_df_sorted=trade_history_df.sort_values(by='exit_time')
         equity_curve_data=equity_history[-(len(trade_history_df_sorted)):]
         equity_curve_s=pd.Series(equity_curve_data, index=trade_history_df_sorted['exit_time'])
         equity_curve_s=equity_curve_s[~equity_curve_s.index.duplicated(keep='last')].sort_index()
         stats=calculate_statistics(trade_history_df_sorted, equity_curve_s, initial_equity)
         try:
             fig, ax = plt.subplots(figsize=(12, 6))
             ax.plot(equity_curve_s.index, equity_curve_s.values, label='Equity Curve', marker='.', linestyle='-')
             ax.set_title('Backtest Equity Progression'); ax.set_xlabel('Time'); ax.set_ylabel('Equity ($)')
             ax.grid(True); ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
         except Exception as plot_err: st.warning(f"Avertissement plot: {plot_err}"); fig=None
    else:
         equity_curve_s=pd.Series(dtype=float)
         stats=calculate_statistics(trade_history_df, equity_curve_s, initial_equity)
    return trade_history_df, equity_curve_s, stats, fig


# ==============================================================
# Fonction Plot Single Trade (INCHANGÉE)
# ==============================================================
# (Identique à la version précédente - non recopiée ici pour la brièveté)
def plot_single_trade(csv_filepath, trade_info, params):
    """Identique à la version précédente"""
    try:
        entry_time = trade_info['entry_time']; exit_time = trade_info['exit_time']
        entry_price = trade_info['entry_price']; exit_price = trade_info['exit_price']
        stop_loss = trade_info['stop_loss']; take_profit = trade_info['take_profit']
        trade_type = trade_info['entry_type']
        time_buffer=pd.Timedelta(minutes=60); plot_start_time=entry_time-time_buffer; plot_end_time=exit_time+time_buffer
        df_full=pd.read_csv(csv_filepath, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, errors='coerce'))
        df_full=df_full.set_index('Date').sort_index()
        plot_df=df_full[(df_full.index >= plot_start_time) & (df_full.index <= plot_end_time)].copy()
        if plot_df.empty: st.warning("Pas de données pour cette fenêtre."); return None
        for col in ['Open', 'High', 'Low', 'Close']: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        if plot_df.empty: st.warning("Données OHLC invalides pour cette fenêtre."); return None
        plot_df['ema_short']=ta.trend.ema_indicator(plot_df['Close'],window=params['ema_short_period'])
        plot_df['ema_long']=ta.trend.ema_indicator(plot_df['Close'],window=params['ema_long_period'])
        plot_df['rsi']=ta.momentum.rsi(plot_df['Close'],window=params['rsi_length'])
        plot_df['atr']=ta.volatility.average_true_range(plot_df['High'],plot_df['Low'],plot_df['Close'],window=params['atr_period'])
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.6,0.2,0.2])
        fig.add_trace(go.Candlestick(x=plot_df.index,open=plot_df['Open'],high=plot_df['High'],low=plot_df['Low'],close=plot_df['Close'],name='OHLC'),row=1,col=1)
        fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['ema_short'],mode='lines',name=f'EMA({params["ema_short_period"]})',line=dict(color='blue',width=1)),row=1,col=1)
        fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['ema_long'],mode='lines',name=f'EMA({params["ema_long_period"]})',line=dict(color='orange',width=1)),row=1,col=1)
        lec,ltp,lsl,lex = 'grey','green','red','purple'; pos_right = "bottom right"; pos_left="top right"
        fig.add_hline(y=entry_price,line_dash="dash",line_color=lec,annotation_text="Entrée",annotation_position=pos_right,row=1,col=1)
        fig.add_hline(y=take_profit,line_dash="dot",line_color=ltp,annotation_text="TP",annotation_position=pos_right,row=1,col=1)
        fig.add_hline(y=stop_loss,line_dash="dot",line_color=lsl,annotation_text="SL",annotation_position=pos_left,row=1,col=1)
        fig.add_hline(y=exit_price,line_dash="dashdot",line_color=lex,annotation_text="Sortie",annotation_position=pos_left,row=1,col=1)
        lemc="blue"; fig.add_vline(x=entry_time,line_width=1,line_dash="dash",line_color=lemc); fig.add_vline(x=exit_time,line_width=1,line_dash="dash",line_color=lemc)
        ms = 'triangle-up' if trade_type=='long' else 'triangle-down'; mc = 'green' if trade_type=='long' else 'red'; mec='purple'
        fig.add_trace(go.Scatter(x=[entry_time],y=[entry_price],mode='markers',name='Entrée Pt',marker=dict(symbol=ms,color=mc,size=12)),row=1,col=1)
        fig.add_trace(go.Scatter(x=[exit_time],y=[exit_price],mode='markers',name='Sortie Pt',marker=dict(symbol='x',color=mec,size=10)),row=1,col=1)
        fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['rsi'],mode='lines',name='RSI',line=dict(color='purple',width=1)),row=2,col=1)
        fig.add_hline(y=params['rsi_overbought'],line_dash="dash",line_color="red",row=2,col=1); fig.add_hline(y=params['rsi_oversold'],line_dash="dash",line_color="green",row=2,col=1)
        fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df['atr'],mode='lines',name='ATR',line=dict(color='grey',width=1)),row=3,col=1)
        if params['atr_threshold']>0: fig.add_hline(y=params['atr_threshold'],line_dash="dot",line_color="blue",name='Seuil ATR',row=3,col=1)
        fig.update_layout(title=f"Visualisation Trade #{trade_info.name} ({trade_type.upper()})",xaxis_rangeslider_visible=False,height=800,showlegend=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        fig.update_yaxes(title_text="Prix",row=1,col=1); fig.update_yaxes(title_text="RSI",range=[0,100],row=2,col=1); fig.update_yaxes(title_text="ATR",row=3,col=1)
        return fig
    except Exception as e: st.error(f"Erreur génération graphique trade: {e}"); import traceback; st.error(traceback.format_exc()); return None


# ==============================================================
# --- Interface Utilisateur Streamlit (MODIFIÉE pour choix Spread) ---
# ==============================================================

st.set_page_config(layout="wide", page_title="Backtester Stratégie v1.9")
st.title("Backtester Stratégie (Choix SL/Spread, Filtres, Concurrence, Visualisation)")

# --- Barre Latérale (Sidebar) ---
st.sidebar.header("Paramètres du Backtest")
csv_file = st.sidebar.text_input("Chemin Fichier CSV", "BTC-USD_data_1min.csv", help="Colonnes: 'Date', 'Open', 'High', 'Low', 'Close'.")
initial_equity = st.sidebar.number_input("Capital Initial ($)", min_value=1.0, value=5000.0, step=100.0, format="%.2f")

# --- MODIFIÉ: Choix Type Spread et Inputs Conditionnels ---
st.sidebar.subheader("Coûts de Transaction")
spread_mode_input = st.sidebar.radio(
    "Type de Spread",
    ["Fixe ($)", "Pourcentage (%)"],
    key='spread_mode',
    help="Comment simuler le spread à l'entrée."
)

spread_cost_val = 0.0
spread_pct_val = 0.0
spread_pct_input_raw = 0.0 # Pour affichage

if spread_mode_input == "Fixe ($)":
    spread_cost_val = st.sidebar.number_input(
        "Spread Fixe ($)",
        min_value=0.0, value=60.0, step=0.5, format="%.2f",
        help="Coût fixe ajouté/soustrait du prix d'entrée."
    )
    spread_type_arg = "fixed"
    # spread_pct_val reste 0.0
else: # Pourcentage (%)
    spread_pct_input_raw = st.sidebar.number_input(
        "Spread (%)",
        min_value=0.00, max_value=5.0, value=0.05, step=0.01, format="%.3f",
        help="Ex: 0.05 pour 0.05%. Ajouté/soustrait au prix d'entrée."
    )
    # Convertir le % entré par l'utilisateur en décimal pour les calculs
    spread_pct_val = spread_pct_input_raw / 100.0
    spread_type_arg = "percentage"
    # spread_cost_val reste 0.0
# --- FIN MODIFICATION ---


st.sidebar.subheader("Indicateurs Techniques")
# ... (inputs EMA, RSI, ATR - INCHANGÉS) ...
ema_short = st.sidebar.number_input("Période EMA Courte", min_value=2, max_value=5000, value=50, step=1, format="%d", help="Inférieure à EMA Longue.")
ema_long = st.sidebar.number_input("Période EMA Longue", min_value=3, max_value=10000, value=200, step=1, format="%d", help="Supérieure à EMA Courte.")
if ema_long <= ema_short: st.sidebar.error("EMA Longue doit être > EMA Courte.")
rsi_len = st.sidebar.number_input("Période RSI", min_value=2, max_value=100, value=14, step=1, format="%d")
rsi_os = st.sidebar.number_input("Seuil RSI Oversold", min_value=1, max_value=50, value=30, step=1, format="%d", help="Inférieur à Overbought.")
rsi_ob = st.sidebar.number_input("Seuil RSI Overbought", min_value=50, max_value=99, value=70, step=1, format="%d", help="Supérieur à Oversold.")
if rsi_ob <= rsi_os: st.sidebar.error("RSI Overbought doit être > RSI Oversold.")
atr_period_input = st.sidebar.number_input("Période ATR", min_value=2, max_value=100, value=14, step=1, format="%d", help="Période pour l'ATR.")


st.sidebar.subheader("Gestion du Risque et Filtres")
# ... (input Risque % - INCHANGÉ) ...
risk_pct_input = st.sidebar.number_input("Risque par Trade (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f", help="Pourcentage du capital actuel à risquer.")
risk_pct = risk_pct_input / 100.0

# ... (Choix Type SL et inputs conditionnels - INCHANGÉ) ...
sl_mode_input = st.sidebar.radio("Type de Stop Loss", ["Pourcentage (%)", "ATR"], key='sl_mode', help="Méthode de calcul du SL.")
sl_pct_input = 0.0; atr_multiplier_sl_input = 0.0
if sl_mode_input == "Pourcentage (%)":
    sl_pct_input = st.sidebar.number_input("Stop Loss (%)", min_value=0.01, max_value=10.0, value=0.2, step=0.01, format="%.2f", help="SL en %.")
    sl_type_arg = "percentage"; sl_pct = sl_pct_input / 100.0; atr_multiplier_sl_input = 2.0
else: # ATR
    atr_multiplier_sl_input = st.sidebar.number_input("Multiplicateur ATR (SL)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, format="%.1f", help="SL = Multiplicateur * ATR.")
    sl_type_arg = "atr"; sl_pct = 0.002

# ... (input TP Multiplier - INCHANGÉ) ...
tp_mult = st.sidebar.number_input("Multiplicateur TP (x Risque)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f", help="TP = Multiplicateur * Distance_SL.")

# ... (input Filtre ATR & Checkbox Concurrence - INCHANGÉ) ...
st.sidebar.subheader("Filtres et Concurrence")
atr_filter_threshold = st.sidebar.number_input("ATR Min. pour Trader", min_value=0.0, value=0.0, step=0.01, format="%.5f", help="Si > 0, ignore signaux si ATR < seuil.")
one_trade_at_a_time_input = st.sidebar.checkbox("Limiter à un seul trade ouvert", value=True, help="Coché: 1 trade max. Décoché: multi-trades.")

# --- Bouton de Lancement ---
params_valid = (ema_long > ema_short) and (rsi_ob > rsi_os)
run_button = st.sidebar.button("Lancer le Backtest", disabled=not params_valid)
st.sidebar.markdown("---"); st.sidebar.info("Backtester v1.9 (Choix Spread)")


# --- Zone d'Affichage Principale ---
st.header("Résultats du Backtest")
# ... (Logique Session State - INCHANGÉE) ...
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False; st.session_state.trade_history = pd.DataFrame()
    st.session_state.equity_curve = pd.Series(dtype=float); st.session_state.statistics = {}
    st.session_state.equity_fig = None; st.session_state.backtest_params = {}

if run_button:
    # Préparation des infos pour l'affichage et l'appel
    concurrency_mode = "Unique" if one_trade_at_a_time_input else "Multiple"
    spread_info = f"{spread_cost_val:.2f}$" if spread_type_arg == "fixed" else f"{spread_pct_input_raw:.3f}%"
    info_str = f"Lancement backtest (SL: {sl_mode_input}, Spr: {spread_info}"
    if atr_filter_threshold > 0: info_str += f", ATR > {atr_filter_threshold:.5f}"
    info_str += f", Conc: {concurrency_mode}) sur '{csv_file}'..."
    st.info(info_str)
    progress_placeholder_area = st.empty()

    current_params = { # Stocke les paramètres pour la visualisation
        "ema_short_period": ema_short, "ema_long_period": ema_long,
        "rsi_length": rsi_len, "rsi_oversold": rsi_os, "rsi_overbought": rsi_ob,
        "atr_period": atr_period_input, "atr_threshold": atr_filter_threshold,
    }

    # Appel à backtest_strategy avec les nouveaux paramètres de spread
    th, ec, stats, efig = backtest_strategy(
        csv_filepath=csv_file, initial_equity=initial_equity,
        ema_short_period=ema_short, ema_long_period=ema_long,
        rsi_length=rsi_len, rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
        risk_percentage=risk_pct,
        sl_type=sl_type_arg, stop_loss_percentage=sl_pct,
        atr_period=atr_period_input, atr_multiplier_sl=atr_multiplier_sl_input,
        atr_threshold=atr_filter_threshold,
        # --- Passage des paramètres Spread ---
        spread_type=spread_type_arg,
        spread_cost=spread_cost_val,
        spread_percentage=spread_pct_val, # La valeur DÉCIMALE est passée
        # --- Fin Passage ---
        take_profit_multiplier=tp_mult,
        progress_placeholder=progress_placeholder_area,
        one_trade_at_a_time=one_trade_at_a_time_input
    )

    progress_placeholder_area.empty()

    # Sauvegarde dans Session State (INCHANGÉ)
    st.session_state.trade_history = th; st.session_state.equity_curve = ec
    st.session_state.statistics = stats; st.session_state.equity_fig = efig
    st.session_state.backtest_params = current_params; st.session_state.results_calculated = True

# --- Affichage des Résultats et Section Visualisation Trade (INCHANGÉ) ---
if st.session_state.results_calculated:
    stats = st.session_state.statistics; equity_fig = st.session_state.equity_fig
    trade_history = st.session_state.trade_history
    if stats:
        st.subheader("Période de Trading")
        first_date=stats.get('First Trade Date'); last_date=stats.get('Last Trade Date'); date_format='%Y-%m-%d %H:%M:%S'; date_col1, date_col2 = st.columns(2)
        with date_col1: st.markdown(f"**Premier Trade:**"); st.write(first_date.strftime(date_format) if first_date else "N/A")
        with date_col2: st.markdown(f"**Dernier Trade:**"); st.write(last_date.strftime(date_format) if last_date else "N/A")
        st.divider()
        st.subheader("Statistiques Clés")
        col1,col2,col3=st.columns(3); col1.metric("Profit ($)",f"{stats.get('Total Profit',0):,.2f}"); col2.metric("Profit (%)",f"{stats.get('Profit (%)',0):.2f}%"); col3.metric("PF",f"{stats.get('Profit Factor',0):.2f}" if stats.get('Profit Factor')!=float('inf') else "Inf")
        col4,col5,col6=st.columns(3); col4.metric("Trades",f"{stats.get('Number of Trades',0):,}"); col5.metric("Win (%)",f"{stats.get('Winning Trades (%)',0):.2f}%"); col6.metric("Max DD (%)",f"{stats.get('Max Drawdown (%)',0):.2f}%")
        col7,col8,col9=st.columns(3); col7.metric("Max Loss Str",f"{stats.get('Max Consecutive Losing Trades',0)}"); col8.metric("Avg Loss Str",f"{stats.get('Average Consecutive Losing Trades',0):.1f}"); col9.metric("Final Eq ($)",f"{stats.get('Final Equity',0):,.2f}")
        st.divider()
        st.subheader("Courbe d'Équité")
        if equity_fig: st.pyplot(equity_fig)
        elif stats.get('Number of Trades', 0) > 0 : st.warning("Graphique Equité non généré.")
        st.subheader("Historique des Trades (5 premiers et 5 derniers)")
        if not trade_history.empty:
            st.dataframe(pd.concat([trade_history.head(),trade_history.tail()]).style.format({"entry_price":"{:.5f}","exit_price":"{:.5f}","stop_loss":"{:.5f}","take_profit":"{:.5f}","profit":"{:,.2f}","size":"{:.4f}"}))
            st.download_button(label="Télécharger l'hist. (CSV)",data=trade_history.to_csv(index=False).encode('utf-8'),file_name='trade_history.csv',mime='text/csv')
        elif stats.get('Number of Trades',-1)==0: st.info("Aucun trade exécuté.")
        st.divider()
        st.subheader("Visualisation d'un Trade Spécifique")
        if not trade_history.empty:
            max_trade_index = len(trade_history) - 1
            selected_trade_index = st.number_input(f"Index du trade (0 à {max_trade_index})", min_value=0, max_value=max_trade_index, value=0, step=1, key='trade_selector')
            if st.button("Afficher le Graphique du Trade", key='show_trade_btn'):
                 with st.spinner("Génération graphique trade..."):
                     trade_details = trade_history.iloc[selected_trade_index]
                     backtest_params = st.session_state.backtest_params # Utilise params sauvegardés
                     single_trade_fig = plot_single_trade(csv_file, trade_details, backtest_params)
                     if single_trade_fig: st.plotly_chart(single_trade_fig, use_container_width=True)
                     else: st.warning("Impossible d'afficher graphique trade.")
        else: st.info("Aucun trade à visualiser.")
    elif not st.session_state.trade_history.empty: st.error("Erreur récupération stats.")
elif not params_valid: st.warning("Corriger erreurs paramètres (EMA/RSI).")
else: st.info("Configurer paramètres et lancer le backtest.")
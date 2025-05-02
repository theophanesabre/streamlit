# -*- coding: utf-8 -*-
# softbacktest.py
# Backtester adapted for Dual EMA / RSI strategy + ML Filter + File Uploads + Dynamic Features
# Current Date: 2025-05-02

import streamlit as st
import pandas as pd
import ta
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import traceback
import numpy as np
import datetime
import joblib
from io import BytesIO

# Configuration Matplotlib et Page Streamlit
matplotlib.use('Agg')
st.set_page_config(layout="wide", page_title="Backtester - Dual EMA/RSI + Uploads")

# --- Injection CSS et Titre ---
st.markdown("""
<style>
    /* Base styles */
    body { font-family: 'Helvetica Neue', sans-serif; }
    div.main { background-color: #000000 !important; color: #FFFFFF !important; }
    div[data-testid="stAppViewContainer"], .stApp { color: #FFFFFF !important; }
    /* Input labels */
    div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label,
    div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label,
    div[data-testid="stToggle"] label, div[data-testid="stSelectbox"] label,
    div[data-testid="stDateInput"] label { color: #EEEEEE !important; }
    /* Metric cards */
    div[data-testid="stMetric"] { background-color: #111111 !important; border-radius: 5px; padding: 10px; color: #FFFFFF !important; }
    div[data-testid="stMetric"] > label { color: #AAAAAA !important; }
    div[data-testid="stMetric"] > div:nth-of-type(2) { color: #FFFFFF !important; }
    /* Dataframes */
    .stDataFrame { color: #333; }
    /* Headings */
    h1[data-testid="stHeading"], .stApp > header h1 { font-size: 2.5rem !important; }
</style>
""", unsafe_allow_html=True)
st.title("Backtester - Stratégie Dual EMA/RSI (Spread Réaliste + Filtre ML)")
# --- Fin Style ---

# ==============================================================
# Fonctions
# ==============================================================

def calculate_statistics(trade_history, equity_curve, initial_equity, equity_history_list):
    """Calcule les statistiques clés du backtest."""
    # (Identique)
    stats = {}
    number_of_trades = len(trade_history)
    stats['Number of Trades'] = number_of_trades; stats['First Trade Date'] = None; stats['Last Trade Date'] = None
    stats['Total Profit'] = 0; stats['Final Equity'] = initial_equity; stats['Profit (%)'] = 0; stats['Winning Trades (%)'] = 0
    stats['Max Drawdown (%)'] = 0; stats['Max Consecutive Losing Trades'] = 0; stats['Average Consecutive Losing Trades'] = 0
    stats['Average Profit per Trade'] = 0; stats['Profit Factor'] = 0
    if number_of_trades == 0: stats['Final Equity'] = initial_equity; stats['Max Drawdown (%)'] = 0; return stats
    if not isinstance(trade_history, pd.DataFrame): trade_history_df = pd.DataFrame(trade_history)
    else: trade_history_df = trade_history
    if trade_history_df.empty: stats['Final Equity'] = initial_equity; stats['Max Drawdown (%)'] = 0; return stats
    try:
        if 'entry_time' in trade_history_df.columns and not pd.api.types.is_datetime64_any_dtype(trade_history_df['entry_time']): trade_history_df['entry_time'] = pd.to_datetime(trade_history_df['entry_time'], errors='coerce')
        if 'exit_time' in trade_history_df.columns and not pd.api.types.is_datetime64_any_dtype(trade_history_df['exit_time']): trade_history_df['exit_time'] = pd.to_datetime(trade_history_df['exit_time'], errors='coerce')
        if 'entry_time' in trade_history_df.columns: trade_history_df.dropna(subset=['entry_time'], inplace=True)
        if 'exit_time' in trade_history_df.columns: trade_history_df.dropna(subset=['exit_time'], inplace=True)
        if trade_history_df.empty: stats['Final Equity'] = initial_equity; stats['Max Drawdown (%)'] = 0; return stats
        trade_history_df_sorted = trade_history_df.sort_values(by='exit_time'); stats['First Trade Date'] = trade_history_df_sorted['entry_time'].iloc[0]; stats['Last Trade Date'] = trade_history_df_sorted['exit_time'].iloc[-1]; total_profit = trade_history_df_sorted['profit'].sum(); stats['Total Profit'] = total_profit; final_equity = initial_equity + total_profit; stats['Final Equity'] = final_equity; stats['Profit (%)'] = (total_profit / initial_equity) * 100 if initial_equity > 0 else 0; winning_trades = trade_history_df_sorted[trade_history_df_sorted['profit'] > 0]; losing_trades = trade_history_df_sorted[trade_history_df_sorted['profit'] <= 0]; stats['Winning Trades (%)'] = len(winning_trades) / number_of_trades * 100 if number_of_trades > 0 else 0
        max_drawdown_value = 0.0
        if equity_history_list and len(equity_history_list) > 1:
            equity_values = pd.Series(equity_history_list); peak = equity_values.iloc[0]
            for equity_val in equity_values:
                if equity_val > peak: peak = equity_val
                if peak > 0: drawdown = (peak - equity_val) / peak; max_drawdown_value = max(max_drawdown_value, drawdown)
            max_drawdown_value = max(0.0, max_drawdown_value)
        stats['Max Drawdown (%)'] = max_drawdown_value * 100
        losing_streak, max_losing_streak, losing_streak_lengths = 0, 0, [];
        for profit in trade_history_df_sorted['profit']:
            if profit <= 0: losing_streak += 1
            else:
                if losing_streak > 0: losing_streak_lengths.append(losing_streak); max_losing_streak = max(max_losing_streak, losing_streak); losing_streak = 0
        if losing_streak > 0: losing_streak_lengths.append(losing_streak); max_losing_streak = max(max_losing_streak, losing_streak)
        stats['Max Consecutive Losing Trades'] = max_losing_streak; stats['Average Consecutive Losing Trades'] = sum(losing_streak_lengths) / len(losing_streak_lengths) if losing_streak_lengths else 0; stats['Average Profit per Trade'] = total_profit / number_of_trades if number_of_trades > 0 else 0; gross_profit = winning_trades['profit'].sum(); gross_loss = abs(losing_trades['profit'].sum()); stats['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    except Exception as e: st.error(f"Err stats: {e}"); st.error(traceback.format_exc()); default_stats = {k: 0 for k in ['Number of Trades','Total Profit','Profit (%)','Winning Trades (%)','Max Drawdown (%)','Max Consecutive Losing Trades','Average Consecutive Losing Trades','Average Profit per Trade','Profit Factor']}; default_stats['Final Equity'] = initial_equity; return default_stats
    return stats

@st.cache_data
def load_data_and_indicators(file_input, ema_short_p, ema_long_p, rsi_p, calc_adx=False, adx_p=14, calculate_ml_features=False, atr_p=14):
    """ Charge données, nettoie, calcule indicateurs et features ML avec noms dynamiques."""
    # (Identique)
    input_name = getattr(file_input, 'name', 'fichier chargé')
    ema_short_col = f'ema_{ema_short_p}'; ema_long_col = f'ema_{ema_long_p}'; price_ema_short_dist_col = f'price_ema_{ema_short_p}_dist_norm'; price_ema_long_dist_col = f'price_ema_{ema_long_p}_dist_norm'; ema_ratio_col = f'ema_ratio_{ema_short_p}_{ema_long_p}'
    active_calcs = [f"EMA({ema_short_p})", f"EMA({ema_long_p})", f"RSI({rsi_p})"];
    if calc_adx: active_calcs.append(f"ADX({adx_p})")
    if calculate_ml_features: active_calcs.append(f"Features ML (ATR({atr_p}), etc.)")
    st.write(f"CACHE MISS/Param Change: Chargement/Prep Données depuis '{input_name}' ({', '.join(active_calcs)})...")
    calculated_indic_cols = []
    try:
        if hasattr(file_input, 'seek'): file_input.seek(0)
        df = pd.read_csv(file_input, encoding='utf-8', on_bad_lines='warn')
        df.rename(columns={'Open time': 'timestamp','Open': 'Open','High': 'High','Low': 'Low','Close': 'Close','Volume': 'Volume'}, inplace=True, errors='ignore')
        required_cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols): raise ValueError(f"Colonnes manquantes: {[c for c in required_cols if c not in df.columns]}.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce'); df.dropna(subset=['timestamp'], inplace=True); df.set_index('timestamp', inplace=True); df = df.sort_index()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        current_rows = len(df); min_rows_needed = 1
        min_rows_needed = max(min_rows_needed, ema_short_p + 1, ema_long_p + 1, rsi_p + 1, atr_p + 1)
        if calc_adx: min_rows_needed = max(min_rows_needed, adx_p * 2)
        if calculate_ml_features: min_required_ml = 60 + 1; min_rows_needed = max(min_rows_needed, min_required_ml)
        if current_rows < min_rows_needed: st.error(f"Données insuffisantes ({current_rows}) / (min: ~{min_rows_needed})"); return pd.DataFrame()
        if ema_short_p < len(df): df[ema_short_col] = ta.trend.ema_indicator(df['Close'], window=ema_short_p); calculated_indic_cols.append(ema_short_col)
        else: df[ema_short_col] = pd.NA
        if ema_long_p < len(df): df[ema_long_col] = ta.trend.ema_indicator(df['Close'], window=ema_long_p); calculated_indic_cols.append(ema_long_col)
        else: df[ema_long_col] = pd.NA
        if rsi_p < len(df): df['rsi'] = ta.momentum.rsi(df['Close'], window=rsi_p); calculated_indic_cols.append('rsi')
        else: df['rsi'] = pd.NA
        if calc_adx and adx_p*2 < len(df):
            if df[['High', 'Low', 'Close']].isnull().any().any(): df['adx'], df['di_pos'], df['di_neg'] = pd.NA, pd.NA, pd.NA
            else:
                try: import pandas_ta as pta; adx_df=df.ta.adx(length=adx_p); adx_col=f'ADX_{adx_p}'; dmp_col=f'DMP_{adx_p}'; dmn_col=f'DMN_{adx_p}'; df['adx']=adx_df[adx_col] if adx_col in adx_df else pd.NA; df['di_pos']=adx_df[dmp_col] if dmp_col in adx_df else pd.NA; df['di_neg']=adx_df[dmn_col] if dmn_col in adx_df else pd.NA; calculated_indic_cols.extend(['adx','di_pos','di_neg'])
                except ImportError:
                    try: df['adx']=ta.trend.adx(df['High'],df['Low'],df['Close'],window=adx_p); df['di_pos']=ta.trend.adx_pos(df['High'],df['Low'],df['Close'],window=adx_p); df['di_neg']=ta.trend.adx_neg(df['High'],df['Low'],df['Close'],window=adx_p); calculated_indic_cols.extend(['adx','di_pos','di_neg'])
                    except Exception as e_ta_adx: st.warning(f"Err ADX 'ta': {e_ta_adx}"); df['adx'],df['di_pos'],df['di_neg'] = pd.NA,pd.NA,pd.NA
                except Exception as e_adx: st.warning(f"Err ADX pta: {e_adx}"); df['adx'],df['di_pos'],df['di_neg'] = pd.NA,pd.NA,pd.NA
        else: df['adx'], df['di_pos'], df['di_neg'] = pd.NA, pd.NA, pd.NA
        ml_feature_columns_generated = []
        if calculate_ml_features:
            print("Calcul features ML...");
            try:
                if atr_p < len(df): df['atr']=ta.volatility.average_true_range(df['High'],df['Low'],df['Close'],window=atr_p); close_numeric=pd.to_numeric(df['Close'],errors='coerce').replace(0,np.nan); df['atr_norm']=df['atr']/close_numeric; df.drop(columns=['atr'],inplace=True,errors='ignore'); ml_feature_columns_generated.append('atr_norm')
                else: df['atr_norm'] = pd.NA
            except Exception as e_atr: print(f"Warn ATR: {e_atr}"); df['atr_norm'] = pd.NA
            if ema_short_col in df.columns: ema_short_numeric=pd.to_numeric(df[ema_short_col],errors='coerce').replace(0,np.nan); df[price_ema_short_dist_col]=(df['Close']-ema_short_numeric)/ema_short_numeric; ml_feature_columns_generated.append(price_ema_short_dist_col)
            else: df[price_ema_short_dist_col] = pd.NA
            if ema_long_col in df.columns: ema_long_numeric=pd.to_numeric(df[ema_long_col],errors='coerce').replace(0,np.nan); df[price_ema_long_dist_col]=(df['Close']-ema_long_numeric)/ema_long_numeric; ml_feature_columns_generated.append(price_ema_long_dist_col)
            else: df[price_ema_long_dist_col] = pd.NA
            if ema_short_col in df.columns and ema_long_col in df.columns: ema_short_numeric=pd.to_numeric(df[ema_short_col],errors='coerce'); ema_long_numeric=pd.to_numeric(df[ema_long_col],errors='coerce').replace(0,np.nan); df[ema_ratio_col]=ema_short_numeric/ema_long_numeric; ml_feature_columns_generated.append(ema_ratio_col)
            else: df[ema_ratio_col] = pd.NA
            try: close_numeric_log=pd.to_numeric(df['Close'],errors='coerce').replace(0,np.nan); df['log_return_1h']=np.log(close_numeric_log/close_numeric_log.shift(60)).fillna(0); ml_feature_columns_generated.append('log_return_1h')
            except Exception as e_logret: print(f"Warn LogRet: {e_logret}"); df['log_return_1h'] = pd.NA
            print("Features ML calculées."); cols_to_check_na = list(set(calculated_indic_cols + ml_feature_columns_generated))
        else: cols_to_check_na = list(set(calculated_indic_cols))
        cols_to_check_na_exist = [col for col in cols_to_check_na if col in df.columns]; essential_price_cols = ['Open', 'High', 'Low', 'Close']
        for p_col in essential_price_cols:
            if p_col not in cols_to_check_na_exist: cols_to_check_na_exist.append(p_col)
        if cols_to_check_na_exist: df.dropna(subset=cols_to_check_na_exist, inplace=True)
        if df.empty: st.error("Aucune ligne valide après nettoyage NaN."); return pd.DataFrame()
        st.write("Fin chargement et préparation."); return df
    except FileNotFoundError: st.error(f"Fichier introuvable: '{input_name}'"); return pd.DataFrame()
    except ValueError as e: st.error(f"Erreur chargement (ValueError): {e}"); return pd.DataFrame()
    except Exception as e: st.error(f"Erreur chargement/préparation: {e}"); st.error(traceback.format_exc()); return pd.DataFrame()

def backtest_strategy(df_processed, initial_equity=5000,
                      ema_short_period=50, ema_long_period=200, rsi_length=14, rsi_oversold=30, rsi_overbought=70,
                      pine_sl_percentage=0.002, pine_tp_multiplier=5.0, sizing_type='risk_pct', pine_risk_percentage=0.005, pine_fixed_lot_size=0.01,
                      spread_percentage=0.0, use_ml_filter=False, ml_model=None, ml_feature_list=None, ml_threshold=0.65,
                      use_adx_filter=False, adx_threshold=25.0, one_trade_at_a_time=True, use_min_profit_points_filter=False, min_profit_points_threshold=0.0,
                      progress_placeholder=None ):
    """ Effectue backtest Dual EMA / RSI avec noms de features dynamiques."""
    # (Identique, avec correction indentation ML filter)
    if df_processed.empty: st.error("DataFrame vide."); return pd.DataFrame(), pd.Series(dtype=float), {}, None, []
    if use_ml_filter and ml_model is None: st.error("Filtre ML activé mais modèle non chargé/fourni."); return pd.DataFrame(), pd.Series(dtype=float), {}, None, []
    if use_ml_filter and ml_feature_list:
        missing_ml_features = [f for f in ml_feature_list if f not in df_processed.columns];
        if missing_ml_features: st.error(f"Features ML manquantes: {missing_ml_features}."); return pd.DataFrame(), pd.Series(dtype=float), {}, None, []
    fig=None; closed_trades_history=[]; equity_history=[initial_equity]; equity=initial_equity; open_positions=[]; trade_id_counter=0; total_rows=len(df_processed)
    spread_method_info = f"Spread ({spread_percentage*100:.4f}%)" if spread_percentage > 0 else "Aucun spread"; ml_info = f"ML (Seuil={ml_threshold:.2f})" if use_ml_filter and ml_model else "ML Inactif"
    st.write(f"Début boucle ({total_rows} bougies) - Strat: EMA({ema_short_period})/EMA({ema_long_period})/RSI({rsi_length}) - Spread: {spread_method_info} - {ml_info}")
    ema_short_col = f'ema_{ema_short_period}'; ema_long_col = f'ema_{ema_long_period}'
    for i, (index, row) in enumerate(df_processed.iterrows()):
        if progress_placeholder and (i % 500 == 0 or i == total_rows - 1): prog = float(i+1)/total_rows; perc = min(int(prog*100),100); progress_placeholder.text(f"Progression: {perc}% ({i+1}/{total_rows})")
        current_close_price = row['Close']
        if pd.isna(current_close_price) or current_close_price <= 0 or pd.isna(row['High']) or pd.isna(row['Low']): continue
        signal_price = row['Close']; current_high = row['High']; current_low = row['Low']; adx_val = row.get('adx', 0)
        positions_to_remove = []; dynamic_spread_amount_exit = current_close_price * spread_percentage;
        if dynamic_spread_amount_exit < 0: dynamic_spread_amount_exit = 0
        for position in open_positions:
            exit_price=None; exit_reason=None; pos_id, pos_entry_price, pos_type, pos_sl, pos_tp, pos_size, pos_entry_time = (position['id'],position['entry_price'],position['type'],position['stop_loss'],position['take_profit'],position['size'],position['entry_time'])
            current_bid_low,current_bid_high = current_low,current_high; current_ask_low=current_low+dynamic_spread_amount_exit; current_ask_high=current_high+dynamic_spread_amount_exit
            if pos_type == 'long':
                if current_bid_low <= pos_sl: exit_price,exit_reason = pos_sl,"SL"
                elif current_bid_high >= pos_tp:
                     if exit_reason != "SL": exit_price,exit_reason = pos_tp,"TP"
            elif pos_type == 'short':
                if current_ask_high >= pos_sl: exit_price,exit_reason = pos_sl,"SL"
                elif current_ask_low <= pos_tp:
                    if exit_reason != "SL": exit_price,exit_reason = pos_tp,"TP"
            if exit_price is not None:
                profit = (exit_price-pos_entry_price)*pos_size if pos_type=='long' else (pos_entry_price-exit_price)*pos_size; equity += profit; equity = max(equity, 0)
                closed_trades_history.append({'trade_id':pos_id,'entry_time':pos_entry_time,'entry_price':pos_entry_price,'entry_type':pos_type,'size':pos_size,'stop_loss':pos_sl,'take_profit':pos_tp,'exit_time':index,'exit_price':exit_price,'profit':profit,'exit_reason':exit_reason})
                equity_history.append(equity); positions_to_remove.append(position)
        for closed_pos in positions_to_remove: open_positions.remove(closed_pos)
        if use_adx_filter and (pd.isna(adx_val) or adx_val < adx_threshold): continue
        final_long_signal=False; final_short_signal=False
        ema_short=row.get(ema_short_col); ema_long=row.get(ema_long_col); rsi_val=row.get('rsi')
        if not pd.isna(signal_price) and not pd.isna(ema_short) and not pd.isna(ema_long) and not pd.isna(rsi_val):
            long_cond = (signal_price>ema_long and ema_short>ema_long and rsi_val<rsi_oversold)
            short_cond = (signal_price<ema_long and ema_short<ema_long and rsi_val>rsi_overbought)
            if long_cond: final_long_signal=True
            elif short_cond: final_short_signal=True
        can_enter = not (one_trade_at_a_time and len(open_positions)>0)
        if can_enter and (final_long_signal or final_short_signal):
            is_long=final_long_signal; is_short=final_short_signal
            if is_long or is_short:
                if equity<=0: st.warning(f"Équité <= 0 à {index}. Arrêt."); break
                # --- CORRECTED ML Filter Block (Indentation + Robust NaN check) ---
                proceed_with_ml = True
                if use_ml_filter and ml_model and ml_feature_list:
                    try:
                        # Indent the following lines under try:
                        features_for_ml_series = row[ml_feature_list]
                        features_for_ml_numeric = pd.to_numeric(features_for_ml_series, errors='coerce')
                        features_for_ml = features_for_ml_numeric.values.reshape(1, -1)
                        if np.isnan(features_for_ml).any():
                            proceed_with_ml = False
                        else:
                            prob_success = ml_model.predict_proba(features_for_ml)[0, 1]
                            if prob_success < ml_threshold:
                                proceed_with_ml = False
                    except KeyError as e_key:
                        st.error(f"Err Feature ML '{e_key}' {index}."); proceed_with_ml = False
                    except Exception as e_ml:
                        st.warning(f"Err ML {index}: {e_ml}."); proceed_with_ml = False
                # --- END CORRECTED ML Filter Block ---
                if proceed_with_ml:
                    base_signal_price=signal_price; actual_entry_price,stop_loss_price,take_profit_price=None,None,None; risk_per_unit,position_size=0.0,0.0
                    if base_signal_price is None or pd.isna(base_signal_price) or base_signal_price<=0: continue
                    try:
                        dynamic_spread_amount_entry=base_signal_price*spread_percentage;
                        if dynamic_spread_amount_entry<0: dynamic_spread_amount_entry=0
                        actual_entry_price=base_signal_price+dynamic_spread_amount_entry if is_long else base_signal_price-dynamic_spread_amount_entry
                        if actual_entry_price<=0: continue
                        sl_factor=(1-pine_sl_percentage) if is_long else (1+pine_sl_percentage); stop_loss_price=actual_entry_price*sl_factor
                        if (is_long and stop_loss_price>=actual_entry_price) or (is_short and stop_loss_price<=actual_entry_price): continue
                        risk_per_unit=abs(actual_entry_price-stop_loss_price);
                        if risk_per_unit<=1e-9: continue
                        tp_offset=risk_per_unit*pine_tp_multiplier; take_profit_price=actual_entry_price+tp_offset if is_long else actual_entry_price-tp_offset
                        if (is_long and take_profit_price<=actual_entry_price) or (is_short and take_profit_price>=actual_entry_price): continue
                        if sizing_type=='risk_pct':
                            if equity<=0: continue
                            risk_amount=equity*pine_risk_percentage; position_size=risk_amount/risk_per_unit
                        elif sizing_type=='fixed_lot': position_size=pine_fixed_lot_size
                        else: continue
                        if position_size<=1e-9: continue
                    except Exception as e: st.warning(f"Err calcul params trade {index}: {e}"); continue
                    execute_this_trade=True
                    if use_min_profit_points_filter and take_profit_price is not None:
                        potential_profit_points=abs(take_profit_price-actual_entry_price);
                        if potential_profit_points < min_profit_points_threshold: execute_this_trade = False
                    if execute_this_trade and position_size > 1e-9 and stop_loss_price is not None and take_profit_price is not None:
                        trade_id_counter+=1; new_position={'id':trade_id_counter,'entry_time':index,'entry_price':actual_entry_price,'type':'long' if is_long else 'short','size':position_size,'stop_loss':stop_loss_price,'take_profit':take_profit_price}; open_positions.append(new_position)
    st.write("Fin boucle backtesting.")
    # --- Final processing - CORRECTED EQUITY CURVE BLOCK ---
    trade_history_df = pd.DataFrame(closed_trades_history)
    equity_curve_s = pd.Series(dtype=float)
    fig = None
    if equity_history:
        equity_dates=[df_processed.index[0]] if not df_processed.empty else [];
        if not trade_history_df.empty and 'exit_time' in trade_history_df.columns: valid_exit_times=pd.to_datetime(trade_history_df['exit_time'],errors='coerce').dropna();
        if 'valid_exit_times' in locals() and not valid_exit_times.empty: equity_dates.extend(valid_exit_times.tolist())
        min_len=min(len(equity_history),len(equity_dates)); equity_history_trimmed=equity_history[:min_len]; equity_dates_trimmed=equity_dates[:min_len]
        try:
            if equity_dates_trimmed:
                 temp_equity_curve=pd.Series(equity_history_trimmed,index=pd.to_datetime(equity_dates_trimmed))
                 if not temp_equity_curve.empty:
                     equity_curve_s=temp_equity_curve[~temp_equity_curve.index.duplicated(keep='last')].sort_index()
        except Exception as eq_curve_err:
            st.warning(f"Erreur création courbe équité: {eq_curve_err}")
            equity_curve_s = pd.Series(dtype=float)
    stats = {};
    if not trade_history_df.empty:
        st.write("Calcul stats & graphiques...");
        try:
            stats=calculate_statistics(trade_history_df,equity_curve_s,initial_equity,equity_history);
            if not equity_curve_s.empty:
                fig,ax=plt.subplots(figsize=(12,6)); ax.plot(equity_curve_s.index,equity_curve_s.values,label='Equity Curve',marker='.',linestyle='-',color='cyan'); ax.set_title('Progression Capital'); ax.set_xlabel('Date'); ax.set_ylabel('Capital ($)'); ax.grid(True,linestyle='--',alpha=0.6); ax.legend(); plt.xticks(rotation=45); plt.tight_layout(); ax.set_facecolor('#111111'); fig.set_facecolor('#000000'); ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white'); ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.title.set_color('white'); ax.tick_params(axis='x',colors='white'); ax.tick_params(axis='y',colors='white'); legend=ax.get_legend()
                if legend:
                    for text in legend.get_texts(): text.set_color("white")
            st.write("Calculs terminés.")
        except Exception as final_calc_err: st.error(f"Err finalisation: {final_calc_err}"); st.error(traceback.format_exc()); equity_curve_s=pd.Series(dtype=float); stats=calculate_statistics(trade_history_df,pd.Series(dtype=float),initial_equity,equity_history); fig=None
    else: stats=calculate_statistics(pd.DataFrame(),pd.Series(dtype=float),initial_equity,[initial_equity]); st.write("Aucun trade.")
    return trade_history_df, equity_curve_s, stats, fig, equity_history

def plot_single_trade(file_input, trade_info, params):
    """ Affiche un graphique détaillé pour un trade spécifique avec noms de features dynamiques."""
    # (Identique, avec correction indentation ADX)
    try:
        entry_time, exit_time, entry_price, exit_price, stop_loss, take_profit, trade_type = (trade_info['entry_time'],trade_info['exit_time'],trade_info['entry_price'],trade_info['exit_price'],trade_info['stop_loss'],trade_info['take_profit'],trade_info['entry_type'])
        entry_time, exit_time = pd.to_datetime(entry_time), pd.to_datetime(exit_time)
        time_diff = exit_time - entry_time if pd.notna(exit_time) and pd.notna(entry_time) else pd.Timedelta(hours=1)
        time_buffer = max(pd.Timedelta(minutes=120), time_diff * 1.5); plot_start_time = entry_time - time_buffer if pd.notna(entry_time) else None; plot_end_time = exit_time + time_buffer if pd.notna(exit_time) else (entry_time + time_buffer * 2 if pd.notna(entry_time) else None)
        if plot_start_time is None or plot_end_time is None: return None
        if file_input is None: st.error("Aucun fichier data pour plot."); return None
        if hasattr(file_input, 'seek'): file_input.seek(0)
        df_full=pd.read_csv(file_input, parse_dates=['Open time'], index_col='Open time', encoding='utf-8', on_bad_lines='skip')
        if not pd.api.types.is_datetime64_any_dtype(df_full.index): df_full.index=pd.to_datetime(df_full.index,errors='coerce'); df_full=df_full.dropna(subset=[df_full.index.name])
        df_full = df_full.sort_index().dropna(subset=['Open','High','Low','Close']); plot_df = df_full.loc[plot_start_time:plot_end_time].copy();
        if plot_df.empty: return None
        for col in ['Open','High','Low','Close']: plot_df[col]=pd.to_numeric(plot_df[col],errors='coerce')
        plot_df.dropna(subset=['Open','High','Low','Close'],inplace=True);
        if plot_df.empty: return None
        ema_short_p = params.get('ema_short_period'); ema_long_p = params.get('ema_long_period'); rsi_p = params.get('rsi_length'); adx_p = params.get('adx_period'); adx_threshold = params.get('adx_threshold')
        ema_short_col = f'ema_{ema_short_p}' if ema_short_p else None
        ema_long_col = f'ema_{ema_long_p}' if ema_long_p else None
        if ema_short_p is not None and ema_short_col and ema_short_p < len(plot_df): plot_df[ema_short_col] = ta.trend.ema_indicator(plot_df['Close'], window=ema_short_p)
        if ema_long_p is not None and ema_long_col and ema_long_p < len(plot_df): plot_df[ema_long_col] = ta.trend.ema_indicator(plot_df['Close'], window=ema_long_p)
        if rsi_p is not None and rsi_p < len(plot_df): plot_df['rsi'] = ta.momentum.rsi(plot_df['Close'], window=rsi_p)
        if adx_p is not None and adx_p*2 < len(plot_df):
            if not plot_df[['High', 'Low', 'Close']].isnull().any().any():
                 try:
                      import pandas_ta as pta
                      adx_df_plot = plot_df.ta.adx(length=adx_p)
                      adx_col_plot=f'ADX_{adx_p}'; dmp_col_plot=f'DMP_{adx_p}'; dmn_col_plot=f'DMN_{adx_p}'
                      if adx_col_plot in adx_df_plot: plot_df['adx']=adx_df_plot[adx_col_plot]
                      else: plot_df['adx']=pd.NA
                      if dmp_col_plot in adx_df_plot: plot_df['di_pos']=adx_df_plot[dmp_col_plot]
                      else: plot_df['di_pos']=pd.NA
                      if dmn_col_plot in adx_df_plot: plot_df['di_neg']=adx_df_plot[dmn_col_plot]
                      else: plot_df['di_neg']=pd.NA
                 except ImportError:
                      try: plot_df['adx']=ta.trend.adx(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p); plot_df['di_pos']=ta.trend.adx_pos(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p); plot_df['di_neg']=ta.trend.adx_neg(plot_df['High'],plot_df['Low'],plot_df['Close'],window=adx_p)
                      except Exception as e_ta_adx_plot: st.warning(f"Err ADX 'ta' plot: {e_ta_adx_plot}"); plot_df['adx'],plot_df['di_pos'],plot_df['di_neg'] = pd.NA,pd.NA,pd.NA
                 except Exception as e_adx_plot: st.warning(f"Err calc ADX plot: {e_adx_plot}"); plot_df['adx'],plot_df['di_pos'],plot_df['di_neg'] = pd.NA,pd.NA,pd.NA
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.65, 0.175, 0.175], subplot_titles=("Prix & Indicateurs", "RSI", "ADX / DI")); fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='OHLC'), row=1, col=1)
        if ema_short_col and ema_short_col in plot_df.columns and not plot_df[ema_short_col].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema_short_col], mode='lines', name=f'EMA({ema_short_p})', line=dict(color='cyan', width=1.5)), row=1, col=1)
        if ema_long_col and ema_long_col in plot_df.columns and not plot_df[ema_long_col].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema_long_col], mode='lines', name=f'EMA({ema_long_p})', line=dict(color='magenta', width=1.5)), row=1, col=1)
        lec, ltp, lsl, lex = 'grey','lime','red','fuchsia'; pos_right="bottom right"; pos_left="top left"; lemc="rgba(100,100,255,0.5)"; ms='triangle-up' if trade_type=='long' else 'triangle-down'; mc='lime' if trade_type=='long' else 'red'; mec='fuchsia'
        if pd.notna(entry_price): fig.add_hline(y=entry_price, line_dash="dash", line_color=lec, annotation_text=f"Entrée {entry_price:.5f}", annotation_position=pos_right, row=1, col=1)
        if pd.notna(take_profit): fig.add_hline(y=take_profit, line_dash="dot", line_color=ltp, annotation_text=f"TP {take_profit:.5f}", annotation_position=pos_right, row=1, col=1)
        if pd.notna(stop_loss): fig.add_hline(y=stop_loss, line_dash="dot", line_color=lsl, annotation_text=f"SL {stop_loss:.5f}", annotation_position=pos_left, row=1, col=1)
        if pd.notna(exit_price): fig.add_hline(y=exit_price, line_dash="dashdot", line_color=lex, annotation_text=f"Sortie {exit_price:.5f}", annotation_position=pos_left, row=1, col=1)
        if pd.notna(entry_time): fig.add_vline(x=entry_time, line_width=1, line_dash="dash", line_color=lemc)
        if pd.notna(exit_time): fig.add_vline(x=exit_time, line_width=1, line_dash="dash", line_color=lemc)
        if pd.notna(entry_time) and pd.notna(entry_price): fig.add_trace(go.Scatter(x=[entry_time], y=[entry_price], mode='markers', name='Entrée Pt', marker=dict(symbol=ms, color=mc, size=12, line=dict(width=1,color='white'))), row=1, col=1)
        if pd.notna(exit_time) and pd.notna(exit_price): fig.add_trace(go.Scatter(x=[exit_time], y=[exit_price], mode='markers', name='Sortie Pt', marker=dict(symbol='x', color=mec, size=10, line=dict(width=1,color='white'))), row=1, col=1)
        if 'rsi' in plot_df.columns and not plot_df['rsi'].isnull().all():
             fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['rsi'], mode='lines', name='RSI', line=dict(color='rgb(180,180,255)', width=1)), row=2, col=1)
             rsi_ob = params.get('rsi_overbought'); rsi_os = params.get('rsi_oversold')
             if rsi_ob is not None: fig.add_hline(y=rsi_ob, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"OB({rsi_ob})", annotation_position="bottom right")
             if rsi_os is not None: fig.add_hline(y=rsi_os, line_dash="dash", line_color="lime", row=2, col=1, annotation_text=f"OS({rsi_os})", annotation_position="bottom right")
        if 'adx' in plot_df.columns and not plot_df['adx'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['adx'], mode='lines', name='ADX', line=dict(color='white', width=1.5)), row=3, col=1)
        if 'di_pos' in plot_df.columns and not plot_df['di_pos'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['di_pos'], mode='lines', name='+DI', line=dict(color='green', width=1)), row=3, col=1)
        if 'di_neg' in plot_df.columns and not plot_df['di_neg'].isnull().all(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['di_neg'], mode='lines', name='-DI', line=dict(color='red', width=1)), row=3, col=1)
        if adx_threshold is not None and adx_threshold > 0: fig.add_hline(y=adx_threshold, line_dash="dot", line_color="aqua", name=f'Seuil ADX ({adx_threshold:.1f})', row=3, col=1)
        fig.update_layout(title=f"Visualisation Trade #{trade_info.name} ({trade_type.upper()})", xaxis_rangeslider_visible=False, height=800, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1), template="plotly_dark")
        fig.update_yaxes(title_text="Prix", row=1, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_yaxes(title_text="RSI", range=[0,100], row=2, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_yaxes(title_text="ADX/DI", row=3, col=1, gridcolor='rgba(180,180,180,0.3)'); fig.update_xaxes(gridcolor='rgba(180,180,180,0.3)')
        return fig
    except FileNotFoundError: return None
    except Exception as e: st.error(f"Err graph trade: {e}"); st.error(traceback.format_exc()); return None

# --- CORRECTED: Indentation error in indicator calculation try/except block ---
def plot_strategy_explanation(ema_short_p, ema_long_p, rsi_p, rsi_os, rsi_ob, sl_pct, tp_rr):
    """ Crée un graphique illustrant les règles avec noms dynamiques."""
    # (Identique à la version précédente, avec correction indentation try/except)
    try:
        required_length = max(ema_short_p, ema_long_p, rsi_p, 50); n_points = required_length+50+int(required_length*0.1); n_points = max(n_points, 200)
        if n_points <= 0: return None
        index=pd.to_datetime(pd.date_range(start='2023-01-01',periods=n_points,freq='h')); np.random.seed(42); base_price=100+np.cumsum(np.random.randn(n_points)*0.5); num_cycles=min(10,n_points/30); price_oscillation=5*np.sin(np.linspace(0,num_cycles*2*np.pi,n_points)); close_np=base_price+price_oscillation; high_np=close_np+np.abs(np.random.randn(n_points))*0.6; low_np=close_np-np.abs(np.random.randn(n_points))*0.4; high_np=np.maximum(high_np,close_np); low_np=np.minimum(low_np,close_np); close=pd.Series(close_np,index=index,name='Close'); high=pd.Series(high_np,index=index,name='High'); low=pd.Series(low_np,index=index,name='Low'); open_=(close.shift(1)+low.shift(1))/2; open_.iloc[0]=close.iloc[0]*0.99; open_.name='Open'; open_=np.where(open_>high,high,open_); open_=np.where(open_<low,low,open_); open_=pd.Series(open_,index=index,name='Open'); df_expl=pd.DataFrame({'Open':open_,'High':high,'Low':low,'Close':close}); df_expl.dropna(inplace=True)
        min_required_for_calc = max(ema_short_p, ema_long_p, rsi_p, 2)
        if len(df_expl) < min_required_for_calc: return None
        ema_short_col = f'ema_{ema_short_p}'; ema_long_col = f'ema_{ema_long_p}'
        try:
            # Indent indicator calculations AND the empty check
            df_expl[ema_short_col]=ta.trend.ema_indicator(df_expl['Close'],window=ema_short_p)
            df_expl[ema_long_col]=ta.trend.ema_indicator(df_expl['Close'],window=ema_long_p)
            df_expl['rsi']=ta.momentum.rsi(df_expl['Close'],window=rsi_p)
            df_expl.dropna(subset=[ema_short_col, ema_long_col,'rsi'],inplace=True)
            # Indent this check
            if df_expl.empty:
                return None
        except Exception as e_ind:
             st.warning(f"Err indic fictifs: {e_ind}"); return None
        long_entry_idx, short_entry_idx = None, None
        if not df_expl.empty: long_cond=(df_expl['Close']>df_expl[ema_long_col])&(df_expl[ema_short_col]>df_expl[ema_long_col])&(df_expl['rsi']<rsi_os); short_cond=(df_expl['Close']<df_expl[ema_long_col])&(df_expl[ema_short_col]<df_expl[ema_long_col])&(df_expl['rsi']>rsi_ob); long_indices=df_expl.index[long_cond]; short_indices=df_expl.index[short_cond];
        if not long_indices.empty: long_entry_idx=long_indices[0]
        if not short_indices.empty: short_entry_idx=short_indices[0]
        fig_expl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=("Exemple Conditions Entrée / SL / TP (sans spread)", "RSI")); fig_expl.add_trace(go.Candlestick(x=df_expl.index,open=df_expl['Open'],high=df_expl['High'],low=df_expl['Low'],close=df_expl['Close'],name='Prix'),row=1,col=1); fig_expl.add_trace(go.Scatter(x=df_expl.index,y=df_expl[ema_short_col],mode='lines',name=f'EMA({ema_short_p})',line=dict(color='cyan',width=1.5)),row=1,col=1); fig_expl.add_trace(go.Scatter(x=df_expl.index,y=df_expl[ema_long_col],mode='lines',name=f'EMA({ema_long_p})',line=dict(color='magenta',width=1.5)),row=1,col=1); fig_expl.add_trace(go.Scatter(x=df_expl.index,y=df_expl['rsi'],mode='lines',name='RSI',line=dict(color='rgb(180,180,255)',width=1)),row=2,col=1); fig_expl.add_hline(y=rsi_ob, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"Overbought ({rsi_ob})", annotation_position="top right"); fig_expl.add_hline(y=rsi_os, line_dash="dash", line_color="lime", row=2, col=1, annotation_text=f"Oversold ({rsi_os})", annotation_position="bottom right")
        if long_entry_idx:
             try:
                  entry_price_l=df_expl.loc[long_entry_idx,'Close']; sl_price_l=entry_price_l*(1-sl_pct); risk_unit_l=entry_price_l-sl_price_l
                  if risk_unit_l > 0:
                       tp_price_l=entry_price_l+(risk_unit_l*tp_rr)
                       fig_expl.add_vline(x=long_entry_idx,line_width=1,line_dash="dash",line_color="lime",row=1,col=1)
                       fig_expl.add_trace(go.Scatter(x=[long_entry_idx],y=[entry_price_l],mode='markers',name='Exemple Long',marker=dict(symbol='triangle-up',color='lime',size=12)),row=1,col=1)
                       fig_expl.add_hline(y=sl_price_l,line_dash="dot",line_color="orange",row=1,col=1,annotation_text=f"SL Long ({sl_price_l:.2f})",annotation_position="bottom left")
                       fig_expl.add_hline(y=tp_price_l,line_dash="dot",line_color="fuchsia",row=1,col=1,annotation_text=f"TP Long ({tp_price_l:.2f})",annotation_position="top left")
                       fig_expl.add_annotation(x=long_entry_idx,y=df_expl['High'].max()*1.01,text=f"Cond. Long: P>EMA({ema_long_p}) & EMA({ema_short_p})>EMA({ema_long_p}) & RSI<{rsi_os}",showarrow=False,bgcolor="rgba(0,100,0,0.6)",row=1,col=1,xanchor="left",yanchor="top")
             except Exception as e_plot_l: st.warning(f"Err plot Long: {e_plot_l}")
        if short_entry_idx:
             try:
                  entry_price_s=df_expl.loc[short_entry_idx,'Close']; sl_price_s=entry_price_s*(1+sl_pct); risk_unit_s=sl_price_s-entry_price_s
                  if risk_unit_s > 0:
                       tp_price_s=entry_price_s-(risk_unit_s*tp_rr)
                       fig_expl.add_vline(x=short_entry_idx,line_width=1,line_dash="dash",line_color="red",row=1,col=1)
                       fig_expl.add_trace(go.Scatter(x=[short_entry_idx],y=[entry_price_s],mode='markers',name='Exemple Short',marker=dict(symbol='triangle-down',color='red',size=12)),row=1,col=1)
                       fig_expl.add_hline(y=sl_price_s,line_dash="dot",line_color="orange",row=1,col=1,annotation_text=f"SL Short ({sl_price_s:.2f})",annotation_position="top right")
                       fig_expl.add_hline(y=tp_price_s,line_dash="dot",line_color="fuchsia",row=1,col=1,annotation_text=f"TP Short ({tp_price_s:.2f})",annotation_position="bottom right")
                       fig_expl.add_annotation(x=short_entry_idx,y=df_expl['Low'].min()*0.99,text=f"Cond. Short: P<EMA({ema_long_p}) & EMA({ema_short_p})<EMA({ema_long_p}) & RSI>{rsi_ob}",showarrow=False,bgcolor="rgba(100,0,0,0.6)",row=1,col=1,xanchor="right",yanchor="bottom")
             except Exception as e_plot_s: st.warning(f"Err plot Short: {e_plot_s}")
        fig_expl.update_layout(title="Illustration Règles (Dual EMA/RSI, sans spread)",height=600,template="plotly_dark",showlegend=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)); fig_expl.update_yaxes(title_text="Prix",row=1,col=1,gridcolor='rgba(180,180,180,0.3)'); fig_expl.update_yaxes(title_text="RSI",range=[0,100],row=2,col=1,gridcolor='rgba(180,180,180,0.3)'); fig_expl.update_xaxes(gridcolor='rgba(180,180,180,0.3)'); fig_expl.update_layout(xaxis_rangeslider_visible=False)
        return fig_expl
    except Exception as e_expl: st.error(f"Err graph expl: {e_expl}"); st.error(traceback.format_exc()); return None

def calculate_periodical_returns(trade_history_df):
    """ Calcule la rentabilité par année, mois, jour. """
    # (Identique)
    if not isinstance(trade_history_df, pd.DataFrame) or trade_history_df.empty: return {'yearly': pd.DataFrame(), 'monthly': pd.DataFrame(), 'daily': pd.DataFrame()}
    df = trade_history_df.copy();
    if 'exit_time' not in df.columns or df['exit_time'].isnull().all(): return {'yearly': pd.DataFrame(), 'monthly': pd.DataFrame(), 'daily': pd.DataFrame()}
    if not pd.api.types.is_datetime64_any_dtype(df['exit_time']): df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce'); df.dropna(subset=['exit_time'], inplace=True)
    if df.empty: return {'yearly': pd.DataFrame(), 'monthly': pd.DataFrame(), 'daily': pd.DataFrame()}
    df['Year'] = df['exit_time'].dt.year; df['YearMonth'] = df['exit_time'].dt.to_period('M'); df['Date'] = df['exit_time'].dt.date
    def win_rate(x):
        if not isinstance(x, pd.Series) or x.empty: return 0; wins = (x > 0).sum(); total = len(x); return (wins / total) * 100 if total > 0 else 0
    agg_funcs = {'profit': ['sum', 'mean', 'count', win_rate]}
    def format_results(df_agg):
        if df_agg.empty: return df_agg; df_agg.columns = ['Total Profit', 'Avg Profit/Trade', 'Nb Trades', 'Win Rate %']; df_agg['Win Rate %'] = df_agg['Win Rate %'].round(2); df_agg['Total Profit'] = df_agg['Total Profit'].round(2); df_agg['Avg Profit/Trade'] = df_agg['Avg Profit/Trade'].round(2); return df_agg
    try: yearly_returns = df.groupby('Year').agg(agg_funcs); yearly_returns = format_results(yearly_returns)
    except Exception as e: st.warning(f"Err rent Ann: {e}"); yearly_returns = pd.DataFrame()
    try:
        monthly_returns = df.groupby('YearMonth').agg(agg_funcs); monthly_returns = format_results(monthly_returns)
        if not monthly_returns.empty: monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
    except Exception as e: st.warning(f"Err rent Mens: {e}"); monthly_returns = pd.DataFrame()
    try: daily_returns = df.groupby('Date').agg(agg_funcs); daily_returns = format_results(daily_returns)
    except Exception as e: st.warning(f"Err rent Jour: {e}"); daily_returns = pd.DataFrame()
    return {'yearly': yearly_returns, 'monthly': monthly_returns, 'daily': daily_returns}

# ==============================================================
# --- Interface Utilisateur Streamlit ---
# ==============================================================

# --- Initialize Session State ---
if 'uploaded_data_file' not in st.session_state: st.session_state.uploaded_data_file = None
if 'uploaded_ml_file' not in st.session_state: st.session_state.uploaded_ml_file = None
default_results_state = {'results_calculated':False, 'trade_history':pd.DataFrame(), 'equity_curve':pd.Series(dtype=float), 'statistics':{}, 'equity_fig':None, 'backtest_params':{}, 'raw_equity_history': [], 'periodical_returns': {}}
for key, value in default_results_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Sidebar ---
st.sidebar.header("Paramètres Backtest Dual EMA/RSI")
st.sidebar.subheader("Fichier de Données Historiques")
uploaded_data_file = st.sidebar.file_uploader("Charger Fichier Données (.csv)", type=['csv'], key='data_uploader')
if uploaded_data_file is not None: st.session_state.uploaded_data_file = uploaded_data_file; st.sidebar.success(f"'{uploaded_data_file.name}' chargé.")
elif st.session_state.uploaded_data_file is not None: st.sidebar.info(f"Utilisera: '{st.session_state.uploaded_data_file.name}'")
else: st.sidebar.warning("Veuillez charger un fichier CSV.")
st.sidebar.divider()
initial_equity = st.sidebar.number_input("Capital Initial ($)", min_value=1.0, value=5000.0, step=100.0, format="%.2f", key="init_eq_main")
st.sidebar.subheader("Période de Backtest")
default_start_date = datetime.date(2017, 1, 1); default_end_date = datetime.date.today()

# --- CORRECTED: Date Input Syntax ---
col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date_input = st.date_input("Date Début", value=default_start_date, key="start_date", max_value=default_end_date)
with col_date2:
    end_date_input = st.date_input("Date Fin", value=default_end_date, key="end_date", min_value=start_date_input)
# --- END CORRECTION ---

dates_valid = True;
if start_date_input > end_date_input: st.sidebar.error("Date début > Date fin."); dates_valid = False
spread_percentage_input = st.sidebar.number_input("Spread (%)", min_value=0.0, value=0.062, step=0.001, format="%.4f", key="spread_percentage")
strategy_type_arg = 'dual_ema_rsi'
st.sidebar.subheader("Indicateurs Stratégie Dual EMA/RSI")
with st.sidebar.container(border=True):
    # --- CORRECTED: EMA Input Syntax ---
    col_ema1, col_ema2 = st.columns(2)
    with col_ema1:
        ema_short_period_input = st.number_input("Période EMA Courte", min_value=2, value=50, step=1, format="%d", key="ema_short")
    with col_ema2:
        ema_long_period_input = st.number_input("Période EMA Longue", min_value=5, value=200, step=1, format="%d", key="ema_long")
    # --- END CORRECTION ---

    rsi_len_param = st.number_input("Période RSI", min_value=2, value=14, step=1, format="%d", key="rsi_len_new")

    # --- CORRECTED: RSI Input Syntax ---
    col_rsi1, col_rsi2 = st.columns(2)
    with col_rsi1:
        rsi_os_val = st.number_input("RSI Oversold", min_value=1, max_value=50, value=30, step=1, format="%d", key="rsi_os_new")
    with col_rsi2:
        rsi_ob_val = st.number_input("RSI Overbought", min_value=50, max_value=99, value=70, step=1, format="%d", key="rsi_ob_new")
    # --- END CORRECTION ---

    # --- CORRECTED: Parameter Validation Logic ---
    params_indic_valid = True # Start assuming valid
    if ema_short_period_input >= ema_long_period_input:
        params_indic_valid = False
        st.sidebar.error("EMA Courte < EMA Longue")
    if rsi_ob_val <= rsi_os_val:
        params_indic_valid = False
        st.sidebar.error("RSI OB > RSI OS")
    # --- END CORRECTION ---

st.sidebar.subheader("Gestion Risque & Sortie")
with st.sidebar.container(border=True):
    st.caption("SL & TP"); pine_sl_pct_input = st.sidebar.number_input("Stop Loss (%)", min_value=0.01, value=0.2, step=0.01, format="%.2f", key="pine_sl_pct"); pine_tp_mult_input = st.sidebar.number_input("Ratio RR", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="pine_rr")
    st.caption("Sizing"); sizing_mode_input = st.radio("Méthode Sizing", ["Risque % Capital", "Lot Fixe"], index=0, key='sizing_mode_new', horizontal=True); sizing_type_arg = 'risk_pct' if sizing_mode_input == 'Risque % Capital' else 'fixed_lot'; pine_risk_pct_val = 0.005; pine_fixed_lot_val = 0.01
    if sizing_type_arg == 'risk_pct': pine_risk_pct_val = st.number_input("Risque %", min_value=0.01, value=0.5, step=0.01, format="%.2f", key="pine_risk_pct") / 100.0
    else: pine_fixed_lot_val = st.number_input("Lot Fixe", min_value=0.0001, value=0.01, step=0.001, format="%.4f", key="pine_fixed_lot")
st.sidebar.subheader("Filtres Additionnels")
with st.sidebar.container(border=True):
    use_adx_filter_input = st.toggle("Filtre ADX", value=False, key="toggle_adx_filter"); adx_period_input = 14; adx_threshold_input = 25.0; calc_adx_for_load = use_adx_filter_input
    # --- CORRECTED: ADX Input Syntax ---
    if use_adx_filter_input:
        col_adx1,col_adx2=st.columns(2)
        with col_adx1:
            adx_period_input=st.number_input("Période ADX", min_value=2, value=14, key="adx_p_filt")
        with col_adx2:
            adx_threshold_input=st.number_input("Seuil ADX", min_value=0.0, value=25.0, step=0.1, format="%.1f", key="adx_thresh_filt")
    # --- END CORRECTION ---
    use_min_profit_filter_input = st.toggle("Filtre Min Profit Pts", value=False, key="toggle_min_profit"); min_profit_points_input = 0.0
    if use_min_profit_filter_input: min_profit_points_input = st.number_input("Seuil Min Profit", min_value=0.0, value=50.0, step=1.0, format="%.5f", key="min_profit_val")
    st.sidebar.markdown("---"); st.sidebar.markdown("**Filtre Machine Learning**")
    uploaded_ml_file = st.sidebar.file_uploader("Charger Modèle (.pkl)", type=['pkl'], key='ml_uploader')
    if uploaded_ml_file is not None: st.session_state.uploaded_ml_file = uploaded_ml_file; st.sidebar.success(f"'{uploaded_ml_file.name}' sélectionné.")
    elif st.session_state.uploaded_ml_file is not None: st.sidebar.info(f"Utilisera '{st.session_state.uploaded_ml_file.name}'")
    else: st.sidebar.info("Aucun modèle chargé.")
    can_enable_ml = st.session_state.uploaded_ml_file is not None; use_ml_filter_input = st.toggle("Activer Filtre ML", value=False, key="toggle_ml_filter", disabled=(not can_enable_ml)); ml_threshold_input = 0.65
    if use_ml_filter_input and can_enable_ml: ml_threshold_input = st.slider("Seuil Prob ML", 0.50, 0.99, 0.65, 0.01, key="ml_threshold")
st.sidebar.subheader("Concurrence Trades"); one_trade_at_a_time_input = st.checkbox("Limiter à 1 trade ouvert", value=True)
run_button_disabled = (st.session_state.uploaded_data_file is None) or (not params_indic_valid) or (not dates_valid)
run_button = st.sidebar.button("🚀 Lancer le Backtest", disabled=run_button_disabled, use_container_width=True)
st.sidebar.markdown("---"); st.sidebar.info("Backtester v3.5.9 - FinalFix UI")

# ==============================================================
# --- Main Display Area ---
# ==============================================================
st.header("Résultats du Backtest")
with st.expander("🔍 Explication Visuelle Stratégie", expanded=False):
    fig_expl = None
    if params_indic_valid: fig_expl=plot_strategy_explanation(ema_short_p=ema_short_period_input,ema_long_p=ema_long_period_input,rsi_p=rsi_len_param,rsi_os=rsi_os_val,rsi_ob=rsi_ob_val,sl_pct=pine_sl_pct_input/100.0,tp_rr=pine_tp_mult_input)
    if fig_expl: st.plotly_chart(fig_expl, use_container_width=True)
    elif not params_indic_valid: st.warning("Params indicateurs invalides.")
    else: st.warning("Err graph expl.")

if run_button:
    if st.session_state.uploaded_data_file is None: st.error("Fichier données non chargé.")
    else:
        for key, value in default_results_state.items(): st.session_state[key] = value
        backtest_can_run = True; ml_model_loaded = None
        if use_ml_filter_input:
            if st.session_state.uploaded_ml_file is not None:
                try: uploaded_file_obj=st.session_state.uploaded_ml_file; uploaded_file_obj.seek(0); ml_model_loaded=joblib.load(uploaded_file_obj); st.info(f"Modèle ML '{uploaded_file_obj.name}' chargé.")
                except Exception as e: st.error(f"Err chargement modèle '{st.session_state.uploaded_ml_file.name}': {e}"); backtest_can_run=False
            else: st.error("Filtre ML activé, mais aucun modèle chargé !"); backtest_can_run=False
        if backtest_can_run:
            # (Le reste du code pour lancer le backtest et afficher les résultats est identique)
            strat_display_name=f"EMA({ema_short_period_input})/EMA({ema_long_period_input})/RSI({rsi_len_param},{rsi_os_val}/{rsi_ob_val})"
            filter_info=[];
            if use_adx_filter_input: filter_info.append(f"ADX({adx_period_input})>{adx_threshold_input:.1f}")
            if use_min_profit_filter_input: filter_info.append(f"MinPtsTP>{min_profit_points_input:.5f}")
            if use_ml_filter_input and ml_model_loaded is not None: filter_info.append(f"ML(>{ml_threshold_input:.2f})")
            filter_str=", ".join(filter_info) if filter_info else "Aucun"; sizing_info=f"{pine_risk_pct_val*100:.2f}% Risk" if sizing_type_arg=='risk_pct' else f"{pine_fixed_lot_val:.4f} Lot"; sl_info=f"{pine_sl_pct_input:.2f}%"; rr_info=f"{pine_tp_mult_input:.1f}"; spread_percentage_decimal=spread_percentage_input/100.0; spread_info=f"Spread:{spread_percentage_input:.4f}%"; concurrency_mode="Unique" if one_trade_at_a_time_input else "Multiple"; info_str=f"Lancement: {strat_display_name} | Sizing:{sizing_info} | SL:{sl_info} | RR:{rr_info} | {spread_info} | Filtres:[{filter_str}] | Conc:{concurrency_mode}"; st.info(info_str); progress_placeholder_area = st.empty()
            current_params = {"strategy_type": strategy_type_arg, "ema_short_period": ema_short_period_input, "ema_long_period": ema_long_period_input, "rsi_length": rsi_len_param, "rsi_oversold": rsi_os_val, "rsi_overbought": rsi_ob_val, "adx_period": adx_period_input if calc_adx_for_load else None, "adx_threshold": adx_threshold_input if use_adx_filter_input else None, "atr_period": 14 }
            st.session_state.backtest_params = current_params
            st.write(f"Préparation données depuis '{st.session_state.uploaded_data_file.name}'...")
            current_data_file = st.session_state.uploaded_data_file; current_data_file.seek(0)
            df_full_with_indicators = load_data_and_indicators(file_input=current_data_file, ema_short_p=ema_short_period_input, ema_long_p=ema_long_period_input, rsi_p=rsi_len_param, calc_adx=calc_adx_for_load, adx_p=adx_period_input, calculate_ml_features=use_ml_filter_input, atr_p = current_params["atr_period"])
            if not df_full_with_indicators.empty:
                st.write("Données chargées. Filtrage dates...")
                start_datetime=pd.to_datetime(start_date_input); end_datetime=pd.to_datetime(end_date_input)+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)
                try:
                    df_filtered = df_full_with_indicators.loc[start_datetime:end_datetime].copy()
                    if df_filtered.empty: st.warning(f"Aucune donnée pour période"); st.session_state.results_calculated = False
                    else:
                        st.write(f"Données filtrées prêtes ({len(df_filtered)} lignes).")
                        with st.spinner("Backtest en cours..."):
                            ml_features_list_for_backtest = None
                            if use_ml_filter_input and ml_model_loaded is not None:
                                try: # Get & check feature names
                                    expected_features = None
                                    if hasattr(ml_model_loaded,'feature_names_in_'): expected_features=list(ml_model_loaded.feature_names_in_)
                                    elif hasattr(ml_model_loaded,'feature_name_'): expected_features=ml_model_loaded.feature_name_()
                                    if expected_features:
                                         ml_features_list_for_backtest=expected_features; available_features=df_filtered.columns.tolist(); missing_model_features=[f for f in expected_features if f not in available_features]
                                         if missing_model_features: st.error(f"Incohérence Features ML: {missing_model_features}"); raise ValueError("Incohérence Features ML.")
                                    else: st.warning("Impossible d'extraire features modèle ML.")
                                except Exception as e_feat: st.warning(f"Err nom features ML: {e_feat}")
                            th,ec,stats,efig,raw_eq_hist = backtest_strategy(df_processed=df_filtered,initial_equity=initial_equity,ema_short_period=ema_short_period_input,ema_long_period=ema_long_period_input,rsi_length=rsi_len_param,rsi_oversold=rsi_os_val,rsi_overbought=rsi_ob_val,pine_sl_percentage=pine_sl_pct_input/100.0,pine_tp_multiplier=pine_tp_mult_input,sizing_type=sizing_type_arg,pine_risk_percentage=pine_risk_pct_val,pine_fixed_lot_size=pine_fixed_lot_val,spread_percentage=spread_percentage_decimal,use_ml_filter=use_ml_filter_input,ml_model=ml_model_loaded,ml_feature_list=ml_features_list_for_backtest,ml_threshold=ml_threshold_input,use_adx_filter=use_adx_filter_input,adx_threshold=adx_threshold_input,one_trade_at_a_time=one_trade_at_a_time_input,use_min_profit_points_filter=use_min_profit_filter_input,min_profit_points_threshold=min_profit_points_input,progress_placeholder=progress_placeholder_area)
                        progress_placeholder_area.empty();
                        periodical_returns_data = calculate_periodical_returns(th)
                        st.session_state.update({'trade_history':th,'equity_curve':ec,'statistics':stats,'equity_fig':efig,'raw_equity_history':raw_eq_hist,'results_calculated':True,'periodical_returns':periodical_returns_data})
                        st.success("Backtest terminé!")
                except Exception as filter_err: st.error(f"Erreur filtrage/backtest: {filter_err}"); st.error(traceback.format_exc()); st.session_state.results_calculated = False
            else: st.error("Chargement données échoué."); st.session_state.results_calculated = False

# --- Display Results Area ---
if st.session_state.results_calculated:
    stats=st.session_state.statistics; equity_fig=st.session_state.equity_fig; trade_history=st.session_state.trade_history; periodical_returns=st.session_state.periodical_returns
    if stats and isinstance(stats, dict) and 'Final Equity' in stats:
        # (Display logic identical)
        st.subheader("Période Trading"); first_trade_dt=pd.to_datetime(stats.get('First Trade Date')); last_trade_dt=pd.to_datetime(stats.get('Last Trade Date')); date_format='%Y-%m-%d %H:%M:%S'; date_col1,date_col2=st.columns(2); date_col1.markdown(f"**Premier Trade:**"); date_col1.write(first_trade_dt.strftime(date_format) if pd.notna(first_trade_dt) else "N/A"); date_col2.markdown(f"**Dernier Trade:**"); date_col2.write(last_trade_dt.strftime(date_format) if pd.notna(last_trade_dt) else "N/A"); st.caption(f"Période sélectionnée: {start_date_input.strftime('%Y-%m-%d')} à {end_date_input.strftime('%Y-%m-%d')}"); st.divider()
        st.subheader("Statistiques Clés"); col1,col2,col3=st.columns(3); col1.metric("Profit Total ($)",f"{stats.get('Total Profit',0):,.2f}"); col2.metric("Profit Total (%)",f"{stats.get('Profit (%)',0):.2f}%"); pf_val=stats.get('Profit Factor',0); col3.metric("Profit Factor",f"{pf_val:.2f}" if pf_val!=float('inf') else "Inf")
        col4,col5,col6=st.columns(3); col4.metric("Nb Trades",f"{stats.get('Number of Trades',0):,}"); col5.metric("Taux Réussite (%)",f"{stats.get('Winning Trades (%)',0):.2f}%"); col6.metric("Max Drawdown (%)",f"{stats.get('Max Drawdown (%)',0):.2f}%")
        col7,col8,col9=st.columns(3); col7.metric("Max Série Pertes",f"{stats.get('Max Consecutive Losing Trades',0)}"); col8.metric("Moy Série Pertes",f"{stats.get('Average Consecutive Losing Trades',0):.1f}"); col9.metric("Capital Final ($)",f"{stats.get('Final Equity', initial_equity):,.2f}"); st.divider()
        st.subheader("Courbe d'Équité");
        if equity_fig: st.pyplot(equity_fig); plt.close(equity_fig)
        elif stats.get('Number of Trades', 0) > 0 : st.warning("Graphique équité non généré.")
        else: st.info("Aucun trade.")
        st.divider()
        st.subheader("Rentabilité Périodique");
        if periodical_returns and isinstance(periodical_returns, dict):
             if not periodical_returns.get('yearly', pd.DataFrame()).empty:
                 with st.expander("🗓️ Annuelle", expanded=False): df_yearly=periodical_returns['yearly']; st.dataframe(df_yearly.style.format("{:.2f}", subset=['Total Profit','Avg Profit/Trade','Win Rate %']).format("{:,.0f}", subset=['Nb Trades']));
                 if 'Total Profit' in df_yearly.columns: st.bar_chart(df_yearly['Total Profit'])
             else: st.info("Aucune donnée annuelle.")
             if not periodical_returns.get('monthly', pd.DataFrame()).empty:
                 with st.expander("🗓️ Mensuelle", expanded=True): df_monthly=periodical_returns['monthly']; st.dataframe(df_monthly.style.format("{:.2f}", subset=['Total Profit','Avg Profit/Trade','Win Rate %']).format("{:,.0f}", subset=['Nb Trades']))
                 if 'Total Profit' in df_monthly.columns:
                      try: chart_data_monthly=df_monthly.reset_index().rename(columns={'index':'YearMonth'}); st.bar_chart(chart_data_monthly,x='YearMonth',y='Total Profit')
                      except Exception as e_chart_month: st.warning(f"Err graph mensuel: {e_chart_month}")
             else: st.info("Aucune donnée mensuelle.")
             if not periodical_returns.get('daily', pd.DataFrame()).empty:
                 with st.expander("🗓️ Journalière (Extrait)", expanded=False): df_daily=periodical_returns['daily']; st.dataframe(df_daily.head(20).style.format("{:.2f}", subset=['Total Profit','Avg Profit/Trade','Win Rate %']).format("{:,.0f}", subset=['Nb Trades'])); st.caption(f"Affichage 20/{len(df_daily)} jours.")
             else: st.info("Aucune donnée journalière.")
        else: st.info("Aucune donnée périodique.")
        st.divider()
        st.subheader("Historique Trades (Extrait)")
        if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
            display_df=trade_history.copy(); cols_to_show=['entry_time','entry_type','entry_price','size','stop_loss','take_profit','exit_time','exit_price','profit','exit_reason']; display_df=display_df[[col for col in cols_to_show if col in display_df.columns]];
            if 'entry_time' in display_df: display_df['entry_time']=pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
            if 'exit_time' in display_df: display_df['exit_time']=pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
            style_format = {"entry_price":"{:.5f}","exit_price":"{:.5f}","stop_loss":"{:.5f}","take_profit":"{:.5f}","profit":"{:,.2f}","size":"{:.4f}"}
            if len(display_df) > 20: st.dataframe(pd.concat([display_df.head(10),display_df.tail(10)]).style.format(style_format)); st.caption(f"Affichage 20/{len(trade_history)} trades.")
            else: st.dataframe(display_df.style.format(style_format))
            csv_data=trade_history.to_csv(index=False).encode('utf-8'); st.download_button(label="📥 Télécharger historique (CSV)",data=csv_data,file_name='trade_history_dual_ema_rsi.csv',mime='text/csv')
        elif stats.get('Number of Trades', -1) == 0: st.info("Aucun trade exécuté.")
        else: st.warning("Historique trades non disponible.")
        st.divider()
        st.subheader("Visualisation Trade Spécifique")
        if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
            if st.session_state.uploaded_data_file is not None:
                trade_options={f"Trade #{int(row['trade_id'])} ({row['entry_type'].upper()}) @ {pd.to_datetime(row['entry_time']).strftime('%Y-%m-%d %H:%M')}":idx for idx, row in trade_history.iterrows()}
                selected_trade_label=st.selectbox(f"Choisir trade ({len(trade_options)}):", options=list(trade_options.keys()), index=0, key='trade_selector_label_main')
                if selected_trade_label:
                     selected_trade_idx_ui=trade_options[selected_trade_label]
                     if st.button("Afficher Graphique Trade", key='show_trade_btn_main'):
                         with st.spinner("Génération graphique..."):
                             trade_details=trade_history.loc[selected_trade_idx_ui]; backtest_params_for_plot=st.session_state.backtest_params;
                             single_trade_fig=plot_single_trade(st.session_state.uploaded_data_file,trade_details,backtest_params_for_plot)
                             if single_trade_fig: st.plotly_chart(single_trade_fig, use_container_width=True)
                             else: st.warning("Impossible d'afficher graphique.")
            else: st.warning("Impossible d'afficher graph trade car fichier données non chargé.")
        else: st.info("Aucun trade à visualiser.")
    elif run_button: st.warning("Backtest terminé sans stats valides.")
    elif not run_button and st.session_state.uploaded_data_file is None: st.warning("Veuillez charger un fichier données.")
    elif not params_indic_valid: st.warning("Params indicateurs invalides.")
    elif not dates_valid: st.warning("Plage de dates invalide.")
elif not run_button and not st.session_state.results_calculated:
     st.info("Configurez paramètres et lancez backtest. Assurez-vous d'avoir chargé un fichier de données.")
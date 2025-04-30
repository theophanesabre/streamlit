# ==============================================================
# --- Interface Utilisateur Streamlit ---
# ==============================================================

# --- Barre Latérale (Sidebar) ---
st.sidebar.header("Paramètres du Backtest")
DEFAULT_DATA_URL = "https://lovecrash.online/btc-usd_data_1min.csv"
NETLIFY_DATA_URL = os.environ.get("DATA_URL", DEFAULT_DATA_URL)
st.sidebar.caption(f"Données: {NETLIFY_DATA_URL.split('/')[-1]}")
initial_equity = st.sidebar.number_input("Capital Initial ($)", min_value=1.0, value=10000.0, step=100.0, format="%.2f")

# --- Choix de la Stratégie ---
st.sidebar.subheader("Stratégie Principale")
strategy_choice = st.sidebar.radio(
    "Type de Stratégie",
    ('Tendance EMA/RSI + Divergence', 'Reversal (MA + RSI Divergence)', 'Croisement EMA + ADX'),
    index=2, key='strat_choice')

strategy_type_arg = 'trend_divergence'
if strategy_choice == 'Reversal (MA + RSI Divergence)': strategy_type_arg = 'reversal_ma_div'
elif strategy_choice == 'Croisement EMA + ADX': strategy_type_arg = 'ema_cross_adx'
is_trend_div_strategy = (strategy_type_arg == 'trend_divergence')
is_reversal_strategy = (strategy_type_arg == 'reversal_ma_div')
is_ema_cross_strategy = (strategy_type_arg == 'ema_cross_adx')

# --- Paramètres des Indicateurs (Affichage Conditionnel) ---
st.sidebar.subheader("Paramètres des Indicateurs")

# Initialiser les variables avec des valeurs par défaut robustes
ema_s_p = 50; ema_l_p = 200; use_ema_rsi = False
reversal_ma_period_input = 100
rsi_len_param = 14; rsi_os_val = 30; rsi_ob_val = 70
adx_period_val = 14; adx_threshold_specific_input = 25.0
div_lookback_val = 30; use_div_trend_strat = False; use_bull_div = False; use_bear_div = False
atr_period_val = 14 # <--- Donner une valeur par défaut ici

# Section EMA
if is_trend_div_strategy or is_ema_cross_strategy:
    with st.sidebar.container(border=True):
        st.caption("Params EMA")
        if is_trend_div_strategy: use_ema_rsi = st.toggle("Utiliser Signal EMA/RSI", value=True, key="toggle_ema_rsi_trend")
        if (is_trend_div_strategy and use_ema_rsi) or is_ema_cross_strategy:
            col_ema1, col_ema2 = st.columns(2)
            with col_ema1: ema_s_p=st.number_input("EMA Rapide",min_value=2,value=50,step=1,format="%d", key="ema_s")
            with col_ema2: ema_l_p=st.number_input("EMA Longue",min_value=2,value=200,step=1,format="%d", key="ema_l")
            if ema_l_p <= ema_s_p: st.sidebar.warning("EMA Longue devrait être > EMA Courte.")

# Section SMA Reversal
if is_reversal_strategy:
    with st.sidebar.container(border=True):
        st.caption("Params MA Reversal")
        reversal_ma_period_input = st.number_input("Période SMA Reversal", min_value=2, value=100, step=1, format="%d", key="reversal_ma_p")

# Section RSI
if is_trend_div_strategy or is_reversal_strategy:
    with st.sidebar.container(border=True):
        st.caption("Params RSI")
        rsi_len_param=st.number_input("Période RSI",min_value=2,max_value=100,value=14,step=1,format="%d",key="rsi_len")
        if is_trend_div_strategy and use_ema_rsi:
            col_rsi1, col_rsi2 = st.columns(2)
            with col_rsi1: rsi_os_val=st.number_input("RSI Oversold",min_value=1,max_value=50,value=30,step=1,format="%d", key="rsi_os")
            with col_rsi2: rsi_ob_val=st.number_input("RSI Overbought",min_value=50,max_value=99,value=70,step=1,format="%d", key="rsi_ob")
            if rsi_ob_val <= rsi_os_val: st.sidebar.error("RSI Overbought doit être > RSI Oversold.")

# Section ADX (regroupée avec Filtre ADX Général ci-dessous)
use_adx_f = st.sidebar.toggle("Activer Filtre ADX Général", value=False, key="toggle_adx")
adx_needed_ui = is_ema_cross_strategy or use_adx_f
if adx_needed_ui:
    with st.sidebar.container(border=True):
         st.caption("Params ADX")
         adx_period_val=st.number_input("Période ADX",min_value=2,max_value=100,value=14,step=1,format="%d", key="adx_p")
         if is_ema_cross_strategy: adx_threshold_specific_input = st.number_input("Seuil ADX (Strat EMA Cross)",min_value=0.0,max_value=100.0,value=25.0,step=0.1,format="%.1f", key="adx_t_specific")
         if use_adx_f: adx_threshold_filter_input = st.number_input("Seuil ADX (Filtre Général)",min_value=0.0,max_value=100.0,value=25.0,step=0.1,format="%.1f", key="adx_t_filter")
else: adx_threshold_specific_input = 25.0; adx_threshold_filter_input = 25.0

# Section Divergence
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

# Options Spécifiques Tendance/Div
if is_trend_div_strategy:
     # ... (Code pour Points et Validité Signal - omis pour lisibilité) ...
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

# Période ATR (Input affiché seulement si nécessaire)
# Déterminer si ATR sera nécessaire pour SL ou Filtre
atr_filter_threshold = st.sidebar.number_input(
    "ATR Min. pour Trader", min_value=0.0, value=0.0, step=0.01, format="%.5f",
    help="Si > 0, ignore signaux si ATR < seuil.", key="atr_filt_thresh"
)

# Déterminer le type de SL (en fonction de la stratégie)
if is_ema_cross_strategy:
    sl_type_arg = 'percentage'
    st.sidebar.write("Stop Loss: % Prix (forcé pour cette strat)")
    sl_pct_input = st.sidebar.number_input("Stop Loss (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f", key="sl_pct_ema_cross")
    sl_pct = sl_pct_input / 100.0
    atr_multiplier_sl_input = 1.5 # Valeur par défaut non utilisée
    atr_needed_for_sl = False
else:
    sl_mode_input = st.sidebar.radio("Type Stop Loss", ["% Prix", "ATR"], index=1, key='sl_mode')
    sl_pct = 0.0; atr_multiplier_sl_input = 1.5
    if sl_mode_input == "% Prix":
        sl_pct_input = st.sidebar.number_input("Stop Loss (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f")
        sl_type_arg = "percentage"; sl_pct = sl_pct_input / 100.0
        atr_needed_for_sl = False
    else: # ATR
        atr_multiplier_sl_input = st.sidebar.number_input("Multiplicateur ATR (SL)", min_value=0.1, max_value=10.0, value=1.5, step=0.1, format="%.1f")
        sl_type_arg = "atr"
        atr_needed_for_sl = True

# Afficher l'input de période ATR si nécessaire (pour SL ou Filtre)
atr_needed = atr_needed_for_sl or (atr_filter_threshold > 0)
if atr_needed:
    atr_period_val = st.number_input( # La variable est maintenant définie conditionnellement ici
        "Période ATR", min_value=2, max_value=100, value=14, # La valeur par défaut initiale est utilisée
        step=1, format="%d", key="atr_p_risk_unified"
    )
# Si atr_needed est False, atr_period_val gardera sa valeur par défaut (14) définie au début de la section UI

# Sizing (Conditionnel pour EMA Cross ADX)
if is_ema_cross_strategy:
     sizing_type_arg = 'risk_pct'
     st.sidebar.write("Sizing: Risque % (forcé pour cette strat)")
     risk_pct_input = st.sidebar.number_input("Risque par Trade (%)", min_value=0.01, max_value=5.0, value=0.5, step=0.01, format="%.2f", key="risk_pct_ema_cross")
     risk_pct_val = risk_pct_input / 100.0
     fixed_lot_val = 0.01
else:
     sizing_mode_input = st.sidebar.radio( "Méthode Sizing", ["Risque %", "Lot Fixe"], index=0, key='sizing_mode')
     risk_pct_val = 0.0; fixed_lot_val = 0.0; risk_pct_input=0.5
     if sizing_mode_input == "Risque %":
          risk_pct_input = st.sidebar.number_input("Risque par Trade (%)", min_value=0.01, max_value=5.0, value=0.5, step=0.01, format="%.2f")
          sizing_type_arg = "risk_pct"; risk_pct_val = risk_pct_input / 100.0; fixed_lot_val = 0.01
     else:
          fixed_lot_size_input = st.sidebar.number_input("Taille Lot Fixe", min_value=0.0001, value=0.01, step=0.001, format="%.4f")
          sizing_type_arg = "fixed_lot"; fixed_lot_val = fixed_lot_size_input; risk_pct_val = 0.005

# Take Profit
tp_mult = st.sidebar.number_input("Ratio Risque/Rendement (RR)", min_value=0.1, max_value=20.0, value=4.0, step=0.1, format="%.1f", key="rr_ratio")

# --- Filtres Additionnels & Concurrence ---
st.sidebar.subheader("Filtres Additionnels et Concurrence")
# Filtre Min Profit Points
use_min_profit_filter_input = st.sidebar.toggle( "Activer Filtre Min Profit Points", value=False, key="toggle_min_profit")
min_profit_points_input = 0.0
if use_min_profit_filter_input: min_profit_points_input = st.sidebar.number_input( "Seuil Min Profit Points (Distance Entry->TP)", min_value=0.0, value=50.0, step=1.0, format="%.5f", key="min_profit_val")
# Concurrence
one_trade_at_a_time_input = st.sidebar.checkbox("Limiter à un seul trade ouvert", value=True)

# --- Bouton de Lancement ---
params_valid = True
if (is_trend_div_strategy or is_ema_cross_strategy) and ema_l_p <= ema_s_p: params_valid = False

run_button = st.sidebar.button("🚀 Lancer le Backtest", disabled=not params_valid, use_container_width=True)
st.sidebar.markdown("---"); st.sidebar.info("Backtester v2.8")

# --- Zone d'Affichage Principale ---
st.header("Résultats du Backtest")
# Initialisation Session State (identique)
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated=False; st.session_state.trade_history=pd.DataFrame(); st.session_state.equity_curve=pd.Series(dtype=float)
    st.session_state.statistics={}; st.session_state.equity_fig=None; st.session_state.backtest_params={}

if run_button:
    st.session_state.results_calculated = False; st.session_state.trade_history=pd.DataFrame(); st.session_state.equity_curve=pd.Series(dtype=float)
    st.session_state.statistics={}; st.session_state.equity_fig=None

    # --- Détermination Indicateurs à Calculer (Strict) ---
    calc_ema = is_trend_div_strategy or is_ema_cross_strategy
    calc_reversal_ma = is_reversal_strategy
    calc_rsi = is_trend_div_strategy or is_reversal_strategy
    calc_divergence = (is_trend_div_strategy and use_div_trend_strat) or is_reversal_strategy
    calc_atr = atr_needed # Utilise le flag déterminé dans la section UI
    calc_adx = adx_needed_ui # Utilise le flag déterminé dans la section UI

    # Prépare infos pour affichage (MAJ pour EMA Cross)
    strat_display_name = ""; filter_info = [];
    if use_adx_f: filter_info.append(f"Filtre ADX({adx_period_val})>{adx_threshold_filter_input}")
    if atr_filter_threshold > 0: filter_info.append(f"Filtre ATR({atr_period_val})>{atr_filter_threshold:.5f}")
    if use_min_profit_filter_input: filter_info.append(f"Filtre MinPtsTP>{min_profit_points_input:.5f}")
    filter_str = ", ".join(filter_info) if filter_info else "Aucun"

    if strategy_type_arg == 'trend_divergence':
        active_signals = []
        if use_ema_rsi: active_signals.append(f"EMA({ema_s_p}/{ema_l_p})/RSI({rsi_len_param})")
        if use_div_trend_strat:
             div_sig_parts = [];
             if use_bull_div: div_sig_parts.append("Bull");
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
    # Utiliser les valeurs finales déterminées pour SL/Sizing
    sizing_info = f"{risk_pct_input:.2f}% Risk" if sizing_type_arg == "risk_pct" else f"{fixed_lot_val:.4f} Lot"
    sl_info = f"{sl_pct_input:.2f}% Px" if sl_type_arg=='percentage' else f"{atr_multiplier_sl_input:.1f}*ATR({atr_period_val})"
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
        # Utiliser le seuil ADX pertinent pour le plot (soit spécifique, soit filtre)
        "adx_threshold": adx_threshold_specific_input if is_ema_cross_strategy else (adx_threshold_filter_input if use_adx_f else None),
    }
    st.session_state.backtest_params = current_params

    st.write("Préparation données (via cache)...")
    df_preprocessed = load_data_and_indicators(
        url=NETLIFY_DATA_URL,
        calc_ema=calc_ema, short_ema_p=ema_s_p, long_ema_p=ema_l_p,
        calc_rsi=calc_rsi, rsi_p=rsi_len_param,
        calc_atr=calc_atr, atr_p=atr_period_val, # Utilise la valeur définie plus haut
        calc_divergence=calc_divergence, div_lookback_p=div_lookback_val,
        calc_adx=calc_adx, adx_p=adx_period_val,
        calc_reversal_ma=calc_reversal_ma, reversal_ma_p=reversal_ma_period_input
        )
    st.write("Données prêtes pour backtest.")

    if not df_preprocessed.empty:
        with st.spinner("Backtest en cours..."):
            # Récupérer les bonnes valeurs pour les arguments de backtest_strategy
            final_risk_pct = risk_pct_val
            final_sl_pct = sl_pct
            final_tp_mult = tp_mult
            # Passer le bon seuil ADX selon la stratégie
            final_adx_threshold = adx_threshold_specific_input if is_ema_cross_strategy else adx_threshold_filter_input

            th, ec, stats, efig = backtest_strategy(
                df_processed=df_preprocessed, initial_equity=initial_equity,
                strategy_type=strategy_type_arg,
                # Risque/Sizing/SL/TP
                risk_percentage=final_risk_pct, sizing_type=sizing_type_arg, fixed_lot_size=fixed_lot_val,
                sl_type=sl_type_arg, stop_loss_percentage=final_sl_pct, atr_multiplier_sl=atr_multiplier_sl_input,
                take_profit_multiplier=final_tp_mult,
                # Filtres / Concurrence
                atr_threshold=atr_filter_threshold, one_trade_at_a_time=one_trade_at_a_time_input,
                use_adx_filter=use_adx_f, adx_threshold=final_adx_threshold, # Passer le seuil pertinent
                use_min_profit_points_filter=use_min_profit_filter_input, min_profit_points_threshold=min_profit_points_input,
                # Params Indicateurs
                ema_short_period=ema_s_p, ema_long_period=ema_l_p, rsi_length=rsi_len_param,
                rsi_oversold=rsi_os_val, rsi_overbought=rsi_ob_val, atr_period=atr_period_val, adx_period=adx_period_val,
                reversal_ma_period=reversal_ma_period_input, div_lookback_period=div_lookback_val,
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
# (Code identique à la version précédente)
if st.session_state.results_calculated:
    stats=st.session_state.statistics
    equity_fig=st.session_state.equity_fig
    trade_history=st.session_state.trade_history # Devrait être un DataFrame ici

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
        # S'assurer que trade_history est un DataFrame avant d'y accéder
        if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
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
        if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
            max_trade_index = len(trade_history) - 1;
            # Utiliser l'index numérique implicite (0 à N-1) pour la sélection
            trade_indices = list(range(len(trade_history)))
            default_idx = 0 if max_trade_index >= 0 else None
            if default_idx is not None:
                 selected_trade_idx_ui = st.selectbox(f"Choisir l'index du trade à visualiser (0 à {max_trade_index})",
                                                  options=trade_indices, index=default_idx, key='trade_selector_idx')

                 if selected_trade_idx_ui is not None and st.button("Afficher Graphique Trade", key='show_trade_btn'):
                      with st.spinner("Génération graphique du trade..."):
                          # Récupérer la ligne par son index positionnel avec iloc
                          trade_details = trade_history.iloc[selected_trade_idx_ui]
                          backtest_params_for_plot = st.session_state.backtest_params
                          single_trade_fig = plot_single_trade(NETLIFY_DATA_URL, trade_details, backtest_params_for_plot)
                          if single_trade_fig: st.plotly_chart(single_trade_fig, use_container_width=True)
                          else: st.warning("Impossible d'afficher le graphique pour ce trade.")
            else: st.info("Aucun trade disponible pour la sélection.")
        else: st.info("Aucun trade dans l'historique à visualiser.")

    elif isinstance(st.session_state.trade_history, pd.DataFrame) and not st.session_state.trade_history.empty:
        st.error("Erreur lors de la récupération ou du calcul des statistiques, mais historique de trades disponible.")
        st.dataframe(st.session_state.trade_history)
    elif not params_valid:
         st.warning("Certains paramètres sont invalides. Veuillez corriger les erreurs dans la barre latérale.")
    else:
         st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer le Backtest'.")
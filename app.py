from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import os
import statsmodels

# Refinitiv API Key
ek.set_app_key(os.getenv('JULSLEO'))

# Declare/Create Dash App Object
app = Dash(__name__)

# Visual App Layout
app.layout = html.Div([
    # Benchmark: Label + Input Bar
    # html.Div([
    #     html.H2('Benchmark ID', style={'display':'inline-block', 'margin-right':20}),
    #     dcc.Input(id='benchmark-id', type='text', value="IVV", placeholder='Benchmark ID', style={'display':'inline-block','border': '1px solid black'}),
    #     ], style={'display':'inline-block', 'width':'100%'}),
    #
    # # Asset: Label + Input Bar
    # html.Div([
    #     html.H2('Asset ID',style={'display':'inline-block','margin-top':0, 'margin-right':20}),
    #     dcc.Input(id='asset-id', type='text', value="AAPL.O", placeholder='Asset ID', style={'display':'inline-block'}),
    #     ], style={'display':'inline-block', 'width': '100%'}),
    html.Div([
        html.H2('Benchmark ID: ', style={'display':'inline-block', 'margin-right':20}),
        dcc.Input(id='benchmark-id', type='text', placeholder='Benchmark ID', value='IVV', style={'display':'inline-block'}),
        html.H1('   ', style={'display':'inline-block', 'margin-right':20}),
        html.H2('Asset ID: ', style={'display':'inline-block', 'margin-right':20}),
        dcc.Input(id='asset-id', type='text', placeholder='Asset ID', value='AAPL.O', style={'display':'inline-block'})
        ],
        style={'display':'inline-block', 'width':'100%'}),

# Date Picker Range Filter 1
    html.Div([
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=date(2017, 1, 1),
            max_date_allowed=date.today(),
            initial_visible_month=date(2023, 1, 5),
            # end_date=date(2017, 8, 25)
        ),
        html.Div(id='output-container-date-picker-range')
    ]),

    # Query Button 1
    html.Button('QUERY Refinitiv', id='run-query', n_clicks=0),

    # Header
    html.H2('Raw Data from Refinitiv'),

    # Data Table
    dash_table.DataTable(
        id="history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    # Header
    html.H2('Historical Returns'),

    # Data Table
    dash_table.DataTable(
        id="returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    # Header
    html.H2('Alpha & Beta Scatter Plot'),

    # Date Picker Range Filter 2
    html.Div([
        dcc.DatePickerRange(
            id='my-date-picker-range-plot'
        ),
        html.Div(id='output-container-date-picker-range-plot')
    ]),

    # Header Button 2
    html.Button('Update Alpha & Beta Scatter Plot', id='run-query-2', n_clicks=0),

    # Scatter Plot Graph
    dcc.Graph(id="ab-plot"),
    html.P(id='summary-text', children=""),

    # Alpha & Beta
    html.Div([
        html.H2('Alpha (\u03B1): ', style={'display':'inline-block', 'margin-right':20}),
        dcc.Input(id='alpha', type='text', placeholder='Y-Intercept', style={'display':'inline-block'}),
        html.H1('   ', style={'display':'inline-block', 'margin-right':20}),
        html.H2('Beta (\u03B2): ', style={'display':'inline-block', 'margin-right':20}),
        dcc.Input(id='beta', type='text', placeholder='Slope', style={'display':'inline-block'})
        ],
        style={'display':'inline-block', 'width':'100%'})
])

# Update Raw Data Table from Refinitiv
@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
     State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id, start_date, end_date):
    assets = [benchmark_id, asset_id]

    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]


    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        unadjusted_price_history = unadjusted_price_history.fillna(0)
        unadjusted_price_history = unadjusted_price_history.replace(np.nan, 0)

    return (unadjusted_price_history.to_dict('records'))

# Update Historical Returns Table
@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call=True
)
def calculate_returns(history_tbl):
    # Convert dictionary to dataframe
    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    # Coerce to date object
    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])

    # Sort by date, Group by asset
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    return (
        pd.DataFrame({
            'Date': numerator[dte_col].reset_index(drop=True),
            'Instrument': numerator[ins_col].reset_index(drop=True),
            'rtn': np.log(
                (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                        denominator[prc_col] * denominator[spt_col]
                ).reset_index(drop=True)
            )
        }).pivot_table(values='rtn', index='Date', columns='Instrument').to_dict('records')
    )

# Update 1st Date Filter
@app.callback(
    Output('output-container-date-picker-range', 'children'),
    [Input('my-date-picker-range', 'start_date'), Input('my-date-picker-range', 'end_date')]
    )
def update_output(start_date, end_date):
    string_prefix = 'You have selected: '
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: ' + end_date_string
    if len(string_prefix) == len('You have selected: '):
        return 'Select a date to see it displayed here'
    else:
        return string_prefix

# Set Limits on 2nd Date Filter
@app.callback(
    [Output('my-date-picker-range-plot', 'min_date_allowed'),
     Output('my-date-picker-range-plot', 'max_date_allowed')],
    [Input('my-date-picker-range', 'start_date'), Input('my-date-picker-range', 'end_date')]
)
def update_second_filter_container(start_date, end_date):
    return start_date, end_date

# Update 2nd Date Filter
@app.callback(
    Output('output-container-date-picker-range-plot', 'children'),
    [Input('my-date-picker-range', 'start_date'), Input('my-date-picker-range', 'end_date')]
    )
def update_output(start_date, end_date):
    string_prefix = 'You have selected: '
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: ' + end_date_string
    if len(string_prefix) == len('You have selected: '):
        return 'Select a date to see it displayed here'
    else:
        return string_prefix

# Render Scatter Plot, Alpha & Beta
@app.callback(
    [Output('ab-plot', 'figure'), Output('alpha', 'value'), Output('beta', 'value')],
    Input('run-query-2', 'n_clicks'),
    [State('benchmark-id', 'value'),
     State('asset-id', 'value'),
     State('my-date-picker-range-plot', 'start_date'), State('my-date-picker-range-plot', 'end_date'),
     State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date'),
     State('returns-tbl', 'data'),
     State('history-tbl', 'data')],
    prevent_initial_call=True
)
def render_ab_plot(n_clicks, benchmark_id, asset_id, filtered_start, filtered_end, orig_start_date, orig_end_date, returns, history):
    hist_df = pd.DataFrame(history)
    filtered = pd.date_range(start=filtered_start, end=filtered_end)
    res = hist_df[(hist_df['Date'] >= filtered_start) & (hist_df['Date'] <= filtered_end)]

    dt_prc_div_splt = pd.DataFrame(res)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    # Coerce to date object
    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])

    # Sort by date, Group by asset
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    new_hist = pd.DataFrame({
            'Date': numerator[dte_col].reset_index(drop=True),
            'Instrument': numerator[ins_col].reset_index(drop=True),
            'rtn': np.log(
                (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                        denominator[prc_col] * denominator[spt_col]
                ).reset_index(drop=True)
            )
        }).pivot_table(values='rtn', index='Date', columns='Instrument').to_dict('records')

    scat_plot = px.scatter(new_hist, x=benchmark_id, y=asset_id, trendline='ols')
    model = px.get_trendline_results(scat_plot)
    results = model.iloc[0]["px_fit_results"]
    alpha = results.params[0]
    beta = results.params[1]

    rounded_alpha = round(alpha, 3)
    rounded_beta = round(beta, 3)
    return scat_plot, rounded_alpha, rounded_beta


if __name__ == '__main__':
    app.run_server(debug=True)

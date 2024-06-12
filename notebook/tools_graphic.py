import os, sys, glob, json
from itertools import product, compress, chain
from functools import reduce
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import fiona
import rasterio
from rasterio import features
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
#import proplot as pplt
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


def PlotLineCropTS(df, fnid, product_name, season_name, link_ratio, year_all, fn_save):
    # Restacking to add missing values
    df = df.pivot_table(index='year', columns=['fnid','country','name','product','season_name','harvest_month','indicator'], values='value')
    df = df.reindex(index=year_all)
    df = df.T.stack(dropna=False).rename('value').reset_index()
    # Add level
    df['level'] = ''
    fnids_old = link_ratio[fnid].columns
    level_div = sorted(np.unique([fnid[:8] for fnid in fnids_old]))
    year_div = np.sort([int(level[2:6]) for level in level_div])
    for i, (year, level) in enumerate(zip(year_div, level_div)):
        if i == 0:
            df.loc[(df['year'] < year) | (df['year'] >= year), 'level'] = level
        else:
            df.loc[(df['year'] >= year), 'level'] = level
    # Footnote
    grain_code = pd.read_hdf('../data/crop/grain_cpcv2_code.hdf')
    product_category = grain_code[['product', 'product_category']].set_index('product').to_dict()['product_category']
    fnid_link_ratio = link_ratio[fnid].rename(product_category, axis=0)
    fnid_link_ratio = fnid_link_ratio.loc[pd.IndexSlice[product_name, season_name],:].T.squeeze(axis=1).sort_index()
    equation = ''
    if isinstance(fnid_link_ratio, pd.Series):
        for i, (f,r) in enumerate(fnid_link_ratio.iteritems()):
            if i == 0:
                equation += '<br>= %s * %.3f' % (f,r)
            else:
                equation += '<br>+ %s * %.3f' % (f,r)
        footnote = "%s (%s - %s)%s" % (fnid, product_name, season_name, equation)
    else:
        for i, (f,r) in enumerate(fnid_link_ratio.droplevel([0,1],axis=1).iterrows()):
            r = r[r.notna()]
            if i == 0:
                equation += f'<br>= {f} ('
            else:
                equation += f')<br>+ {f} ('
            for j, (c, v) in enumerate(r.iteritems()):
                if j == 0:
                    equation += f'{c}*{v:.3f}'
                else:
                    equation += f' + {c}*{v:.3f}'
        footnote = "%s (%s - %s)%s)" % (fnid, product_name, season_name, equation)
    # Plotting
    fig = px.line(df, x='year', y='value', color='level', markers=True,
                  range_x=list(year_all[[0,-1]]),
                  facet_row='indicator', facet_row_spacing=0.02,
                  category_orders = {'indicator': ['production','area','yield']},
                 )
    fig.update_layout(
        width=700, height=400,
        font=dict(family='arial', size=12, color='black'),
        margin={"r":0,"t":0,"l":0,"b":0},
        annotations=[],
        xaxis=dict(range=year_all[[0,-1]], title={'text': ''}),
        yaxis3=dict(title='Production'),
        yaxis2=dict(title='Area'),
        yaxis=dict(title='Yield'),
        template='plotly',
        # hovermode="x",
        legend=dict(title='',font_size=12,x=1.35,y=1,xanchor='right',yanchor='top',bgcolor='rgba(0,0,0,0)'),
    )
    fig.for_each_annotation(lambda x: x.update(text=''))
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(matches=None, rangemode="tozero")
    fig.update_traces(connectgaps=False)
    fig.add_annotation(
        xref='paper',yref='paper',
        x=1.35, y=0,
        text=footnote,
        align="right",
        yanchor='bottom',
        showarrow=False,
        font = {'family':'arial','size':12, 'color':'dimgrey'},
    )
    fig.add_annotation(
        xref='x domain',yref='y3 domain',x=0.0,y=1,text='(mt)',align="left",showarrow=False,
        font={'family':'arial','size':12, 'color':'dimgrey'},
    )
    fig.add_annotation(
        xref='x domain',yref='y2 domain',x=0.0,y=1,text='(ha)',align="left",showarrow=False,
        font={'family':'arial','size':12, 'color':'dimgrey'},
    )
    fig.add_annotation(
        xref='x domain',yref='y domain',x=0.0,y=1,text='(mt/ha)',align="left",showarrow=False,
        font={'family':'arial','size':12, 'color':'dimgrey'},
    )
    if fn_save:
        fig.write_image(fn_save, scale=1.3)
        print('%s is saved.' % fn_save)
    return fig


def PlotMapCropSystem(df, footnote, fn_save=False):
    shape = gpd.read_file('../data/shapefile/fewsnet/SO_Admin2_1990.shp')
    shape.geometry = shape.geometry.simplify(0.01)
    geojson = json.loads(shape[['FNID','geometry']].to_json())
    df['crop_production_system'].replace({'agro_pastoral': 100,'riverine':10, 'none':1}, inplace=True)
    df = df.pivot_table(index=['season_date','season_name'], columns=['fnid'], values='crop_production_system', aggfunc='sum')/3
    system = (df == 10).sum(0) > 0
    sub = shape.copy()
    sub['riverine'] = 0
    sub.loc[sub['FNID'].isin(system.index[~system]), 'riverine'] = 1
    sub.loc[sub['FNID'].isin(system.index[system]), 'riverine'] = 2
    sub.to_file('../data/shapefile/fewsnet/SO_Admin2_1990_riverine.shp')
    sub = shape.copy()
    sub['riverine'] = 'No-data'
    sub.loc[sub['FNID'].isin(system.index[~system]), 'riverine'] = 'Agro-pastoral'
    sub.loc[sub['FNID'].isin(system.index[system]), 'riverine'] = 'Agro-pastoral+Riverine'
    fig = px.choropleth(
        locations=sub['FNID'],
        color = sub['riverine'],
        color_discrete_sequence=['No-data','Agro-pastoral','Agro-pastoral+Riverine'],
        color_discrete_map={'No-data':'lightgrey',
                            'Agro-pastoral':'lightgreen',
                            'Agro-pastoral+Riverine':'cyan'},
        geojson=geojson,
        featureidkey='properties.FNID',
    )
    fig.update_geos(visible=False, resolution=50,
                    showcountries=True, countrycolor="white",
                    lonaxis_range=[40.5, 51],
                    lataxis_range=[-2, 12],
                    showframe=False,
                   )
    fig.update_layout(
        width=600, height=550,
        margin={"r":0,"t":0,"l":0,"b":0},
        font=dict(family='arial', size=15, color='black'),
        dragmode=False,
        legend=dict(title='Crop production system')
    )
    fig.add_annotation(
            xref='paper',yref='paper',
            x=0.04, y=0.0,
            text=footnote,
            align="left",
            showarrow=False,
            font = {'family':'arial','size':15,'color':'dimgrey'},
        )
    if fn_save:
        fig.write_image(fn_save, scale=1.0)
        print('%s is saved.' % fn_save)
    return fig


def PlotHeatSeasonData(data, code, comb, comb_name, footnote, fn_save=False):
    # Combinations of "seasons"
    years = np.arange(data['year'].min(), data['year'].max()+1)
    data['season_name'].replace(code, inplace=True)
    data = data.pivot_table(index=['year','fnid'], columns=['product'], values='season_name', aggfunc='sum')/3
    data = data.stack().reset_index()
    data.columns = ['year','fnid','product','value']
    # Covert to easy code number
    data['value'].replace(comb,inplace=True)
    # Color palette
    ncomb = len(comb_name)
    bvals = np.arange(ncomb+1)+0.5
    colors = px.colors.qualitative.Plotly[:ncomb]
    dcolorsc = discrete_colorscale(bvals, colors)
    # FNIDs and years for plotting
    fnids = sorted(data['fnid'].unique())
    data = data.pivot_table(index='year',columns='fnid',values='value')
    data = data.reindex(index=years,columns=fnids)
    # Plotting
    fig = go.Figure(
        data=go.Heatmap(
            z=data.astype(str),
            visible=True,
            coloraxis = 'coloraxis',
            hovertemplate = 'FNID: %{x}<br>Year: %{y}<br>System: %{z:d}<extra></extra>',
            dx=1,dy=1
        )
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":20},
        font=dict(family='arial', size=15, color='black'),
        height=800, width=1200, 
        xaxis=dict(
            title='',
            dtick=1,
            tickmode = 'array',
            tickvals = np.arange(data.shape[1]),
            ticktext=data.columns,
            tickfont_size=14
        ),
        yaxis=dict(
            title='',
            autorange='reversed',
            dtick=1,
            tickmode = 'array',
            tickvals = np.arange(len(years)),
            ticktext= years,
            tickfont_size=14
        ),
        coloraxis=dict(
            colorscale=dcolorsc,
            cmin=0.5,
            cmax=ncomb+0.5,
            colorbar=dict(
                x=1.01,
                y=0.5,
                len=0.7,
                thickness=15,
                outlinewidth=1,
                title='Season',
                tickvals=tuple(comb_name.keys()),
                ticktext=tuple(comb_name.values())
            )
        ),   
    )
    fig.add_annotation(
        xref='paper',yref='paper',
        x=-0.0, y= -0.20,
        text=footnote,
        align="left",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    if fn_save:
        fig.write_image(fn_save)
        print('%s is saved.' % fn_save)
    return fig


def PlotHeatCropSystem(data, code, comb, comb_name, footnote, fn_save=False):
    # Combinations of "crop production system"
    years = np.arange(data['year'].min(), data['year'].max()+1)
    data['crop_production_system'].replace(code, inplace=True)
    data = data.pivot_table(index=['year','season_name'], columns=['fnid'], values='crop_production_system', aggfunc='sum')/3
    data = data.stack().reset_index()
    data.columns = ['year','season_name','fnid','value']
    # Covert to easy code number
    data['value'].replace(comb,inplace=True)
    # Color palette
    ncomb = len(comb_name)
    bvals = np.arange(ncomb+1)+0.5
    colors = px.colors.qualitative.Plotly[:ncomb]
    dcolorsc = discrete_colorscale(bvals, colors)
    # FNIDs and years for plotting
    fnids = sorted(data['fnid'].unique())
    data = data.pivot_table(index='year',columns='fnid',values='value')
    data = data.reindex(index=years,columns=fnids)
    # Plotting
    fig = go.Figure(
        data=go.Heatmap(
            z=data.astype(str),
            visible=True,
            coloraxis = 'coloraxis',
            hovertemplate = 'FNID: %{x}<br>Year: %{y}<br>System: %{z:d}<extra></extra>',
            dx=1,dy=1
        )
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":20},
        font=dict(family='arial', size=15, color='black'),
        height=800, width=1200, 
        xaxis=dict(
            title='',
            dtick=1,
            tickmode = 'array',
            tickvals = np.arange(data.shape[1]),
            ticktext=data.columns,
            tickfont_size=14
        ),
        yaxis=dict(
            title='',
            autorange='reversed',
            dtick=1,
            tickmode = 'array',
            tickvals = np.arange(len(years)),
            ticktext= years,
            tickfont_size=14
        ),
        coloraxis=dict(
            colorscale=dcolorsc,
            cmin=0.5,
            cmax=ncomb+0.5,
            colorbar=dict(
                x=1.01,
                y=0.5,
                len=0.7,
                thickness=15,
                outlinewidth=1,
                title='Crop production system',
                tickvals=tuple(comb_name.keys()),
                ticktext=tuple(comb_name.values())
            )
        ),   
    )
    fig.add_annotation(
        xref='paper',yref='paper',
        x=-0.0, y= -0.20,
        text=footnote,
        align="left",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    if fn_save:
        fig.write_image(fn_save)
        print('%s is saved.' % fn_save)
    return fig

def PlotLinePAY(df, year, footnote, fn_save=None):
    # Restacking to add missing values
    df = df.pivot_table(index='year', columns=['fnid','country','name','product','season_name','harvest_month','indicator'], values='value')
    df = df.reindex(index=np.arange(year[0], year[-1]+1))
    df = df.T.stack(dropna=False).reset_index().rename(columns={0: 'value'})
    # Plotting
    indicators = ['production', 'area', 'yield']
    fig, axes = plt.subplots(nrows=len(indicators), ncols=1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.patch.set_facecolor('white')
    for ax, indicator in zip(axes, indicators):
        ax.set_ylabel(f'{indicator.capitalize()} (mt)' if indicator == 'production' else f'{indicator.capitalize()} (ha)' if indicator == 'area' else f'{indicator.capitalize()} (mt/ha)')
        if len(df[df['indicator'] == indicator]) == 0:
            # No data
            continue
        sns.pointplot(
            ax=ax,
            data=df[df['indicator'] == indicator],
            x='year', 
            y='value', 
            hue='fnid',
            linewidth=1.5,
            markers='o',
            markersize=6,
            markeredgecolor='white',
            markeredgewidth=1,
            legend=True
        )
        if indicator != 'production':
            ax.get_legend().remove()
        ax.grid(which='major', axis='both', linestyle='-')
    # Legend settings
    axes[0].legend(bbox_to_anchor=(1.01, 1.03), loc='upper left', frameon=False, labelspacing=0.2)
    # Remove minor xticks
    axes[0].xaxis.set_tick_params(which='both', bottom=False)
    axes[1].xaxis.set_tick_params(which='both', bottom=False)
    axes[2].xaxis.set_tick_params(which='minor', bottom=False)
    # Set xtick labels
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=90)
    # Title and footnote
    ax.set_xlabel('')
    plt.figtext(0.07, 0.015, footnote, wrap=True, horizontalalignment='left', fontsize=11)
    # Layout settings
    plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.12, hspace=0.1)
    plt.show()
    # Save figure
    if fn_save:
        fig.savefig(fn_save, bbox_inches='tight', dpi=100)
        print(f'{fn_save} is saved.')
    return


def PlotBarProduction(df, year, footnote, fn_save=None):
    # Filtering and preparing data
    indicator_exist = df['indicator'].unique()
    indicator_exist = indicator_exist[~np.isin(indicator_exist, 'yield')]
    table = df.pivot_table(
        index='year',          
        columns=['fnid','country','name','product','season_name','harvest_month','indicator'],         
        values='value'
    )
    # National production
    nat = df.groupby(['season_name','product','indicator','year']).sum(min_count=1).reset_index()
    # National production in percentage
    container = []
    for (indicator,season_name) in product(indicator_exist,df.season_name.unique()):
        temp = table.loc[:, pd.IndexSlice[:,:,:,:,season_name,:,indicator]].groupby('product', axis=1).sum(min_count=1)
        temp = temp.div(temp.sum(1), axis=0)*100
        temp = temp.stack().reset_index().rename({0:'value'},axis=1)
        temp['season_name'] = season_name
        temp['indicator'] = indicator
        container.append(temp)
    natp = pd.concat(container, axis=0).reset_index(drop=True)
    natp = natp[['season_name','product','indicator','year','value']]
    # Aggregation
    nat['type'] = 'orig_unit'
    natp['type'] = 'percent'
    both = pd.concat([nat,natp], axis=0)
    both = both[
        (both['indicator'] == 'production') &
        (both['season_name'] == season_name)
    ]
    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    fig.patch.set_facecolor('white')
    # Seasonal production (t)
    data_orig = both[both['type'] == 'orig_unit'].pivot_table(index='year', columns='product', values='value', aggfunc='sum')
    product_rank = data_orig.mean(0).sort_values(ascending=False).index
    data_orig = data_orig.reindex(np.arange(year[0], year[1]+1)).fillna(np.nan)
    data_orig = data_orig[product_rank].sort_index()
    # Seasonal production (%)
    data_percent = both[both['type'] == 'percent'].pivot_table(index='year', columns='product', values='value', aggfunc='sum')
    data_percent = data_percent.reindex(np.arange(year[0], year[1]+1)).fillna(np.nan)
    data_percent = data_percent[product_rank].sort_index()
    # Plotting settings
    ax = axes[0]
    data_orig.plot(ax=ax, kind='bar', stacked=True, width=0.8, colormap='tab20', alpha=1.0, legend=False)
    ax.set_ylabel('Production (t)')
    ax.set_ylim(0, data_orig.sum(1).max()*1.1)
    ax.grid(which='major', axis='both', linestyle='-')
    ax = axes[1]
    data_percent.plot(ax=ax, kind='bar', stacked=True, width=0.8, colormap='tab20', alpha=1.0, legend=False)
    ax.set_ylabel('Production (%)')
    ax.set_ylim(0, 100)
    ax.grid(which='major', axis='both', linestyle='-')
    # Legend settings
    axes[0].legend(bbox_to_anchor=(1.01, 1.03), loc='upper left', frameon=False, labelspacing=0.2)
    # Set xticks
    axes[1].set_xlabel('')
    axes[1].grid(which='minor', axis='x', linestyle='-', color='black', visible=False)
    axes[0].xaxis.set_tick_params(which='both', bottom=False)
    axes[1].xaxis.set_tick_params(which='minor', bottom=False)
    # Title and footnote
    plt.figtext(0.07, 0.015, footnote, wrap=True, horizontalalignment='left', fontsize=11)
    # Layout settings
    plt.subplots_adjust(left=0.07, right=0.85, top=0.95, bottom=0.12, hspace=0.1)
    plt.show()
    # Save figure
    if fn_save:
        fig.savefig(fn_save, bbox_inches='tight', dpi=150)
        print(f'{fn_save} is saved.')
    return


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale
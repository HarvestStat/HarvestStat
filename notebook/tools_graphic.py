import os, sys, glob, json
from itertools import product, compress, chain
from functools import reduce
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')
plt.rcParams['font.sans-serif'] = 'Arial'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


def PlotLinePAY(df, year, footnote, fn_save=None):
    # Restacking to add missing values
    df = df.pivot_table(index='year', columns=['fnid','country','name','product','season_name','harvest_month','indicator'], values='value')
    df = df.reindex(index=np.arange(year[0], year[-1]+1))
    df = df.T.stack(dropna=False).reset_index().rename(columns={0: 'value'})
    if len(df) == 0:
        if footnote:
            print('No data to plot for %s.' % footnote)
        else:
            print('No data to plot.')
        return
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
            legend=True,
            zorder=0
        )
        if indicator != 'production':
            ax.get_legend().remove()
        ax.grid(which='major', axis='both', color='k', linewidth=0.3, linestyle='-', alpha=0.2, zorder=1)
        # Setting frame line width
        for spine in ax.spines.values(): spine.set_linewidth(0.6)
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
        fig.savefig(fn_save, bbox_inches='tight', dpi=300)
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
    for (indicator, season_name) in product(indicator_exist, df.season_name.unique()):
        temp = table.loc[:, pd.IndexSlice[:,:,:,:,season_name,:,indicator]].groupby('product', axis=1).sum(min_count=1)
        temp = temp.div(temp.sum(1), axis=0) * 100
        temp = temp.stack().reset_index().rename({0: 'value'}, axis=1)
        temp['season_name'] = season_name
        temp['indicator'] = indicator
        container.append(temp)
    natp = pd.concat(container, axis=0).reset_index(drop=True)
    natp = natp[['season_name','product','indicator','year','value']]
    # Aggregation
    nat['type'] = 'orig_unit'
    natp['type'] = 'percent'
    both = pd.concat([nat, natp], axis=0)
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
    data_orig = data_orig.reindex(np.arange(year[0], year[1] + 1)).fillna(np.nan)
    data_orig = data_orig[product_rank].sort_index()
    # Seasonal production (%)
    data_percent = both[both['type'] == 'percent'].pivot_table(index='year', columns='product', values='value', aggfunc='sum')
    data_percent = data_percent.reindex(np.arange(year[0], year[1] + 1)).fillna(np.nan)
    data_percent = data_percent[product_rank].sort_index()
    # Plotting settings
    ax = axes[0]
    data_orig.plot(ax=ax, kind='bar', stacked=True, width=0.8, colormap='tab20', alpha=1.0, legend=False, zorder=1)
    ax.set_ylabel('Production (t)')
    ax.set_ylim(0, data_orig.sum(1).max() * 1.1)
    ax.grid(which='major', axis='both', color='k', linewidth=0.3, linestyle='-', alpha=0.2, zorder=1)
    ax = axes[1]
    data_percent.plot(ax=ax, kind='bar', stacked=True, width=0.8, colormap='tab20', alpha=1.0, legend=False, zorder=1)
    ax.set_ylabel('Production (%)')
    ax.set_ylim(0, 100)
    ax.grid(which='major', axis='both', color='k', linewidth=0.3, linestyle='-', alpha=0.2, zorder=1)
    # Legend settings
    axes[0].legend(bbox_to_anchor=(1.01, 1.03), loc='upper left', frameon=False, labelspacing=0.2)
    # Set xticks
    axes[1].set_xlabel('')
    axes[1].grid(which='minor', axis='x', linestyle='-', color='black', visible=False)
    axes[0].xaxis.set_tick_params(which='both', bottom=False)
    axes[1].xaxis.set_tick_params(which='minor', bottom=False)
    # Setting frame line width
    for spine in axes[0].spines.values(): spine.set_linewidth(0.6)
    for spine in axes[1].spines.values(): spine.set_linewidth(0.6)
    # Title and footnote
    plt.figtext(0.07, 0.015, footnote, wrap=True, horizontalalignment='left', fontsize=11)
    # Layout settings
    plt.subplots_adjust(left=0.07, right=0.85, top=0.95, bottom=0.12, hspace=0.1)
    plt.show()
    # Save figure
    if fn_save:
        fig.savefig(fn_save, bbox_inches='tight', dpi=300)
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
import json
from itertools import compress
from functools import reduce
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import rasterio
from rasterio.mask import mask
from rasterio import features
import warnings
import os
import random
import requests
from scipy import ndimage
# Suppress specific RuntimeWarning from shapely
warnings.filterwarnings("ignore", message="invalid value encountered in intersection", category=RuntimeWarning, module="shapely.set_operations")

# Predfined product mapping dictionary; This should cover all product names in the FDW data.
product_name_dict = {
    'African Eggplant': 'Eggplant',
    'African oil palm nut': 'African oil palm nut',
    'Almond (unspecified)': 'Almond',
    'Anise': 'Anise',
    'Apple (unspecified)': 'Apple',
    'Apricot': 'Apricot',
    'Artichoke': 'Artichoke',
    'Avocado (Hass)': 'Avocado (Hass)',
    'Avocado (unspecified)': 'Avocado',
    'Bambara groundnut': 'Bambara groundnut',
    'Banana (unspecified)': 'Banana',
    'Bananas/Plantains, mixed': 'Banana',
    'Barley': 'Barley',
    'Barley (Unspecified)': 'Barley',
    'Basil': 'Basil',
    'Bean (Green, unspecified)': 'Beans (Green)',
    'Bean (Hyacinth)': 'Bean (Hyacinth)',
    'Beans (Lima)': 'Beans (Lima)',
    'Beans (Pinto)': 'Beans (Pinto)',
    'Beans (Red Kidney)': 'Beans (Red Kidney)',
    'Beans (Red)': 'Beans (Red)',
    'Beans (Rosecoco)': 'Beans (Rosecoco)',
    'Beans (White)': 'Beans (White)',
    'Beans (mixed)': 'Beans (mixed)',
    'Beet': 'Beet',
    'Broad Beans': 'Broad Beans',
    'Broccoli': 'Broccoli',
    'Bush bean': 'Bush Bean',
    'Cabbage (Unspecified)': 'Cabbage',
    'Canola seed': 'Canola Seed',
    'Cantaloupe': 'Cantaloupe',
    'Capsicum chinense, unspecified': 'Capsicum Chinense',
    'Carrots': 'Carrots',
    'Cashew (unshelled)': 'Cashew (unshelled)',
    'Cassava': 'Cassava',
    'Cassava (non-bitter)': 'Cassava (non-bitter)',
    'Cauliflowers': 'Cauliflowers',
    'Celery': 'Celery',
    'Cereal Crops (Mixed)': 'Cereal Crops',
    'Champignon': 'Champignon',
    'Chick Peas': 'Chick Peas',
    'Chili pepper (Unspecified)': 'Chili Pepper',
    'Chilles and Peppers': 'Chili Pepper',
    'Clove': 'Clove',
    'Cocoa': 'Cocoa',
    'Cocoyam, move to 1594aa': 'Cocoyam',
    'Coffee (Unspecified)': 'Coffee',
    'Coffee (unspecified)': 'Coffee',
    'Colocynth': 'Colocynth',
    'Cooking Banana (unspecified)': 'Cooking Banana',
    'Coriander': 'Coriander',
    'Cotton (Acala)': 'Cotton (Acala)',
    'Cotton (American)': 'Cotton (American)',
    'Cotton (Egyptian)': 'Cotton (Egyptian)',
    'Cotton (Unspecified)': 'Cotton',
    'Cottonseed (Other)': 'Cottonseed',
    'Cowpea (unspecified)': 'Cowpea',
    'Cowpeas (Mixed)': 'Cowpea',
    'Cucumber': 'Cucumber',
    'Date (unspecified)': 'Date',
    'Eggplant': 'Eggplant',
    'Enset': 'Enset',
    'Ethiopian cabbage': 'Ethiopian Cabbage',
    'Fava bean': 'Fava Bean',
    'Fenugreek': 'Fenugreek',
    'Fibers (unspecified)': 'Fibers',
    'Field Peas': 'Field Peas',
    'Fig (unspecified)': 'Fig',
    'Fodder crop (unspecified)': 'Fodder crop',
    'Fonio': 'Fonio',
    'Garlic': 'Garlic',
    'Garlic (dry)': 'Garlic',
    'Garlic (fresh)': 'Garlic',
    'Geocarpa groundnut': 'Geocarpa groundnut',
    'Gibto': 'Gibto',
    'Ginger': 'Ginger',
    'Gourd (Unspecified)': 'Gourd',
    'Goussi': 'Goussi',
    'Gram (Green)': 'Mung bean',
    'Grape (unspecified)': 'Grape',
    'Grass Pea': 'Grass Pea',
    'Green Peppers': 'Green Peppers',
    'Green bean (fresh)': 'Green Bean',
    'Green pea': 'Green Pea',
    'Groundnut (without shell)': 'Groundnuts (Without Shell)',
    'Groundnuts (In Shell)': 'Groundnuts (In Shell)',
    'Groundnuts (In Shell, Large)': 'Groundnuts (In Shell, Large)',
    'Groundnuts (In Shell, Small)': 'Groundnuts (In Shell, Small)',
    'Guava (unspecified)': 'Guava',
    'Henna': 'Henna',
    'Hops': 'Hops',
    'Hot red pepper': 'Chili Pepper',
    'Jews mallow leaves': 'Molokhia',
    'Jute': 'Jute',
    'Kabuli chick pea': 'Kabuli Chick Pea',
    'Leeks': 'Leeks',
    'Lemon (unspecified)': 'Lemon',
    'Lentils': 'Lentils',
    'Lettuce (Unspecified)': 'Lettuce',
    'Linseed (unspecified)': 'Linseed',
    'Macadamia (unspecified)': 'Macadamia',
    'Maize': 'Maize',
    'Maize (Corn)': 'Maize',
    'Maize Grain (White)': 'Maize',
    'Maize Grain (Yellow)': 'Maize (Yellow)',
    'Mandarin orange': 'Mandarin Orange',
    'Mango (unspecified)': 'Mango',
    'Melon (unspecified)': 'Melon',
    'Millet': 'Millet',
    'Millet (Bulrush)': 'Millet',
    'Millet (Finger)': 'Millet',
    'Millet (Foxtail)': 'Millet',
    'Millet (Pearl)': 'Millet',
    'Mixed Teff': 'Teff',
    'Mung bean (unspecified)': 'Mung bean',
    'Mung bean, n.e.c.': 'Mung bean',
    'Neug': 'Neug',
    'Oats (Unspecified)': 'Oats',
    'Okras (Fresh)': 'Okras',
    'Onions': 'Onions',
    'Orange (unspecified)': 'Orange',
    'Other root/tuber vegetable (unspecified)': 'Other root/tuber vegetable',
    'Other stem vegetables': 'Other stem vegetables',
    'Pam nut or kernal (unspecified)': 'Pam Nut',
    'Papaya (unspecified)': 'Papaya',
    'Paprika (unspecified)': 'Paprika',
    'Pea (unspecified)': 'Pea',
    'Peach (unspecified)': 'Peach',
    'Pepper (Piper spp.)': 'Pepper',
    'Pigeon Peas': 'Pigeon Pea',
    'Pigeon pea (Unspecified)': 'Pigeon Pea',
    'Pineapple (unspecified)': 'Pineapple',
    'Pole bean': 'Pole Bean',
    'Pomegranate': 'Pomegranate',
    'Potato (Irish)': 'Potato',
    'Potato (unspecified)': 'Potato',
    'Pulses, dry, unspecified': 'Pulses (dry)',
    'Quince (unspecified)': 'Quince',
    'Rape': 'Rape',
    'Rape fodder': 'Rape',
    'Rice': 'Rice',
    'Rice (Paddy)': 'Rice',
    'Rice, not husked': 'Rice (not husked)',
    'Safflower Seed': 'Safflower Seed',
    'Sesame Seed': 'Sesame Seed',
    'Sorghum': 'Sorghum',
    'Sorghum (Red)': 'Sorghum (Red)',
    'Sorrel': 'Sorrel',
    'Soybean (unspecified)': 'Soybean',
    'Spanish peanut (in shell)': 'Spanish Peanut',
    'Spinach': 'Spinach',
    'Squash (Pumpkin, Zucchini)': 'Squash',
    'Squash (Unspecified)': 'Squash',
    'Squash and Melon seeds': 'Squash and Melon Seeds',
    'Strawberry (unspecified)': 'Strawberry',
    'Sugarcane (for sugar)': 'Sugarcane',
    'Sunflower Seed': 'Sunflower Seed',
    'Sunflower seed': 'Sunflower Seed',
    'Sweet Potatoes': 'Sweet Potatoes',
    'Sweet Potatoes (Non-Orange)': 'Sweet Potatoes (Non-Orange)',
    'Sweet Potatoes (Orange)': 'Sweet Potatoes (Orange)',
    'Swiss Chard': 'Swiss Chard',
    'Taro (move to 1594AA)': 'Taro',
    'Taro, move to 1594AA': 'Taro',
    'Taro/Cocoyam (Unspecified)': 'Taro',
    'Tea Plant': 'Tea',
    'Tea leaves (Mixed)': 'Tea',
    'Tigernut': 'Tigernut',
    'Tobacco (Burley)': 'Tobacco',
    'Tobacco (unspecified)': 'Tobacco',
    'Tomato': 'Tomato',
    'Tomatoes (Roma, medium)': 'Tomato',
    'Vanilla': 'Vanilla',
    'Vegetables (unspecified)': 'Vegetables',
    'Velvet bean': 'Velvet Bean',
    'Vetch': 'Vetch',
    'Virginia peanut (in shell)': 'Virginia Peanut',
    'Watermelon': 'Watermelon',
    'Wheat': 'Wheat',
    'Wheat Grain': 'Wheat',
    'Yams': 'Yams'
}

def get_product_name_dict():
    return product_name_dict

def product_name_mapping(stack, list_except=None):
    # Here we list all the product names to track the total number of products and their potential changes.
    not_in_custom = [p for p in stack['product'].unique() if p not in product_name_dict.keys()]
    if len(not_in_custom) > 0:
        print(f'Warning: {not_in_custom} are not in the product_name_list.')
    # Exception
    if list_except is not None:
        sub1 = stack[stack['product'].isin(list_except)]
        sub2 = stack[~stack['product'].isin(list_except)]
        sub2['product'] = sub2['product'].replace(product_name_dict)
        stack = pd.concat([sub1, sub2], axis=0).reset_index(drop=True).sort_values(by=['fnid','product','season_name'])
    else:
        stack['product'] = stack['product'].replace(product_name_dict)
    return stack



def load_shapefile(fn, epsg):
    shape = gpd.read_file(fn)
    shape = shape.to_crs(epsg)
    shape['area'] = shape['geometry'].area / 10**6
    return shape

def sort_dict(d):
    return dict(sorted(d.items()))

def invert_dicts(d):
    inverse = {}
    for k, v in d.items():
        inverse[v] = sorted(inverse.get(v, []) + [k])
    return inverse

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = [key]
            else: 
                inverse[item].append(key) 
    return inverse

def save_hdf(filn, df, set_print=True):
    df.to_hdf(filn, key='df', complib='blosc:zstd', complevel=9)
    if set_print:
        print('%s is saved.' % filn)
    return

def save_npz(filn, data, set_print=True):
    np.savez_compressed(filn, data=data)
    if set_print:
        print('%s is saved.' % filn)
    return

def load_npz(filn, key='data'):
    data = np.load(filn, allow_pickle=True)[key].tolist()
    return data


def PrintAdminUnits(shape_all):
    adm_year = shape_all['FNID'].apply(lambda x: str(x)[:8]).value_counts()
    adm_year = adm_year.to_frame().reset_index().rename(columns={'index':'name','FNID':'count'})
    adm_year['year'] = adm_year['name'].apply(lambda x: int(x[2:6]))
    adm1_year = adm_year[adm_year['name'].apply(lambda x: x[-2:] == 'A1')].set_index('year')
    adm2_year = adm_year[adm_year['name'].apply(lambda x: x[-2:] == 'A2')].set_index('year')
    adm3_year = adm_year[adm_year['name'].apply(lambda x: x[-2:] == 'A3')].set_index('year')
    adm_year = pd.merge(adm1_year, adm2_year, left_index=True,right_index=True,how='outer',)
    adm_year = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), [adm1_year, adm2_year, adm3_year])
    adm_year.columns = ['Admin1','# units', 'Admin2', '# units', 'Admin3', '# units']
    adm_year['# units'] = adm_year['# units'].fillna(0).astype(int)
    adm_year = adm_year.sort_index()
    print('- FEWS NET admin shapefiles ------------------- #')
    print('| year\t | Admin1   | # units   | Admin2   | # units   | Admin3   | # units   |')
    for i, (adm1, nadm1, adm2, nadm2, adm3, nadm3) in adm_year.iterrows():
        print('| %d\t | %s | %d\t| %s\t| %d\t| %s\t| %d\t|' % (i, adm1, nadm1, adm2, nadm2, adm3, nadm3))
    print('----------------------------------------------- #')
    return


def FDW_PD_Sweeper(df, area_priority='Area Planted'):
    assert area_priority in ['Area Planted','Area Harvested']
    df_raw = df.copy()
    # Remove missing records ----------------------------- # 
    # Missing data if "status" is "Missing Historic Data" or "Not Collected"
    print('- Remove missing records ---------------------- #')
    print('Orignial data points: {:,}'.format(df.shape[0]))
    # We ignore zero values
    df.loc[df['value'] == 0, 'status'] = 'Zero Value'
    df.loc[df['value'] == 0, 'value'] = np.nan
    # We ignore missing values
    df.loc[df['value'].isna(), 'status'] = 'Missing Value'
    df.loc[df['value'].isna(), 'value'] = np.nan
    cateories, counts = np.unique(df.loc[df['value'].isna(), 'status'],return_counts=True)
    for category, count in zip(cateories, counts):
        print('Removed {:,} "{}" points'.format(count, category))
    for indicator in df['indicator'].unique():
        idx_ind = df['indicator'] == indicator
        idx_nna = df['value'].notna()
        print('{:,}/{:,} "{}" points are retained.'.format(sum(idx_ind & idx_nna), sum(idx_ind), indicator))    
    df = df.loc[df['value'].notna()].reset_index(drop=True)
    print('Current data points: {:,}'.format(df.shape[0]))
    print('')
    assert list(df['status'].unique()) == ['Collected']
    # - Check duplicated data points
    rows = ['fnid','crop_production_system','season_year', 'product','indicator']
    assert df[df[rows].duplicated(keep=False)].shape[0] == 0
    # ---------------------------------------------------- #

    # Minor changes -------------------------------------- #
    # planting year and month (based on FDW's "season_date"; "planting_month" might be manually changed later)
    df['season_date'] = pd.to_datetime(pd.to_datetime(df['season_date']).dt.strftime('%Y-%m-01'))
    df['planting_year'] = df['season_date'].dt.year
    df['planting_month'] = df['season_date'].map(lambda x: '%02d-01' % x.month)
    df['year'] = df['planting_year'].copy()
    # End of harvesting month
    df['harvest_date'] = pd.to_datetime(pd.to_datetime(df['start_date']).dt.strftime('%Y-%m-01'))
    df['harvest_year'] = df['harvest_date'].dt.year
    df['harvest_month'] = df['harvest_date'].map(lambda x: '%02d-01' % x.month)
    print('- Minor changes are applied ------------------- #')
    print('')
    # ---------------------------------------------------- #

    # Basic information ---------------------------------- #
    print('- Basic information --------------------------- #')
    year_min, year_max = df['year'].min(), df['year'].max()
    season_table = df[['season_name', 'season_date']]
    season_table['season_date'] = pd.to_datetime(season_table['season_date']).dt.strftime('%m-01')
    season_table = season_table.drop_duplicates().reset_index(drop=True)
    season_list = list(season_table['season_name']+' ('+season_table['season_date']+')')
    seasons = df['season_name'].unique()
    products = df['product'].unique()
    cps = df['crop_production_system'].unique()
    print("Data period: %d - %d" % (year_min, year_max))
    print("%d grain types are found: %s" % (len(products), ", ".join(sorted(products))))
    print("%d seasons are found: %s" % (len(seasons), ", ".join(season_list)))
    print("%d crop production system are found: %s" % (len(cps), ", ".join(cps)))
    print('Data sources include:')
    sources = df[['source_organization','source_document']].drop_duplicates().reset_index(drop=True)
    for i, (organization, document) in sources.iterrows():
        print('[%d] %s --- %s' % (i+1,organization,document))
    print("Administrative-1 fnids: {:,}".format(sum([t[7] == '1' for t in df.fnid.unique()])))
    print("Administrative-2 fnids: {:,}".format(sum([t[7] == '2' for t in df.fnid.unique()])))
    reporting_unit = df.loc[df['fnid'].map(lambda x: x[6] != 'A'), 'fnid'].unique()
    print('%d reporting units are found: %s' % (len(reporting_unit), ", ".join(reporting_unit)))
    print('')
    # ---------------------------------------------------- #
    
    # Total Production ----------------------------------- #
    print('- Total production over time ------------------ #')
    total_prod = df[df['indicator']=='Quantity Produced'].groupby(['product','season_name'])['value'].sum()
    total_prod_prct = total_prod/total_prod.sum()*100
    total_prod_prct = total_prod_prct.reset_index().pivot_table(index='product',columns='season_name',values='value')
    total_prod_prct[total_prod_prct.isna()] = 0
    total_prod_prct.index.name = ''
    print(total_prod_prct.round(1).astype(str) + '%')
    print('')
    # ---------------------------------------------------- #
    
    # Crop Calendar -------------------------------------- #
    print('- Crop calendar ------------------------------- #')
    temp = df[['product','season_name','planting_month','harvest_month']].drop_duplicates().sort_values('product').reset_index(drop=True)
    print(temp)
    print('')
    # ---------------------------------------------------- #

    # Make data table ------------------------------------ #
    rows = ['fnid','crop_production_system','season_year', 'product']
    cols = ['Area Harvested', 'Area Planted', 'Quantity Produced', 'Yield']
    records = df.pivot_table(index=rows, columns='indicator', values='value', fill_value=np.nan)
    records = records.reindex(columns=cols)
    # Selecting "Area Planted" or "Area Harvested" with area_priorty
    # In either case, the name of area will be "Area Harvested"
    if area_priority == 'Area Planted':
        # Fill missing "Area Planted" with "Area Harvested"
        records["Area Planted"].fillna(records['Area Harvested'], inplace=True)
        records.drop(columns=['Area Harvested'], inplace=True)
        records.rename(columns={'Area Planted':'Area Harvested'}, inplace=True)
    elif area_priority == 'Area Harvested':
        # Fill missing "Area Harvested" with "Area Planted"
        records["Area Harvested"].fillna(records['Area Planted'], inplace=True)
        records.drop(columns=['Area Planted'], inplace=True)
    else:
        raise ValueError('Invalid area_priorty.')
    # Stacking and Merging
    records = records.stack().rename('value').reset_index()
    cols_add = [
        'country_code', 'country','admin_1', 'admin_2', 'admin_3', 'admin_4', 
        'population_group', 'start_date', 'period_date', 
        'season_name','season_type', 'season_date', 
        'planting_year', 'planting_month', 'year', 'harvest_date', 'harvest_year', 'harvest_month',
        'cpcv2', 'cpcv2_description','document_type'
    ]
    # Quick check for no duplicates
    assert df[[*rows]].drop_duplicates().shape[0] == df[[*rows,*cols_add]].drop_duplicates().shape[0]
    df = pd.merge(df[[*rows,*cols_add]].drop_duplicates(), records, on=rows)

    # Record years per season
    print('- Recorded years per season ------------------- #')
    for season in seasons:
        sub = df[df['season_name'] == season]
        years = np.sort(pd.to_datetime(sub['season_date']).dt.year.unique())
        miss = np.setdiff1d(np.arange(years[0],years[-1]+1), years)
        print("%s: %d years (%d-%d) (missing: %d years)" % (season, (years[-1]-years[0]+1), years[0], years[-1], len(miss)))
    print('')
    # Data points per admin boundaries
    print('- Number of "Quantity Produced" data points --- #')
    fnid_short = df['fnid'].apply(lambda x: x[:8])
    for short in sorted(fnid_short.unique()):
        print('{}: {:,} data points are found.'.format(short, sum((fnid_short == short) & (df['indicator'] == 'Quantity Produced'))))
    print('')
    # Districts with specified "population group(s)"
    print('- Districts with population group(s) ---------- #')
    fnid_pop_group = list(df.loc[df['population_group'] != 'none', 'fnid'].unique())
    print("{:,} districts includes 'population_group'.".format(len(fnid_pop_group)))
    if len(fnid_pop_group) > 0:
        print(": %s" % (", ".join(sorted(fnid_pop_group))))
    print('')
    # ---------------------------------------------------- #
    return df, df_raw


def FDW_PD_AvalTable(df, shape_all):
    '''Print the Number of admin units having FDW data in each year.
    '''
    print('Table of available data')
    shape_all['FNID_short'] = shape_all['FNID'].apply(lambda x:x[2:8])
    fnid_dict = shape_all[['FNID_short','FNID']].groupby('FNID_short')['FNID'].apply(list).to_dict()
    temp = df[['year','crop_production_system','product','season_name','fnid']].copy()
    temp['fnid_short'] = temp['fnid'].apply(lambda x: x[2:8])
    temp = temp.groupby(['product','year','season_name','crop_production_system','fnid_short'])['fnid'].apply(list).rename('fnid').reset_index()
    temp['percent'] = np.nan
    for i, row in temp.iterrows():
        if row['fnid_short'][-2] == 'R':
            temp.loc[i, 'percent'] = '-'
            temp.loc[i, 'string'] = '%d' % (len(np.unique(row['fnid'])))
        else:
            temp.loc[i, 'percent'] = len(np.unique(row['fnid']))/len(fnid_dict[row['fnid_short']])*100
            temp.loc[i, 'string'] = '%d/%d' % (len(np.unique(row['fnid'])), len(fnid_dict[row['fnid_short']]))
    table_dict = dict()
    for product_name in temp['product'].unique():
        sub = temp[temp['product'] == product_name]
        sub = sub.pivot_table(index='year',columns=['season_name','crop_production_system','fnid_short'],values='string',aggfunc='first')
        sub = sub.reindex(index=np.arange(sub.index[0],sub.index[-1]+1))
        table_dict[product_name] = sub
    for k, v in table_dict.items():
        print('----------------------------------------------- #')
        print('Crop type: %s' % k)
        print('----------------------------------------------- #')
        print(v.to_string())
    print('----------------------------------------------- #')
    return table_dict

def PlotAdminShapes(shape_plot, label=True):
    shape_plot['level'] = shape_plot['FNID'].apply(lambda x: str(x)[:8])
    levels = shape_plot['level'].unique()
    color = cm.rainbow_r(np.linspace(0, 1, len(levels)))
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),facecolor='w')
    ax.set_axis_off()
    ax.set_aspect('equal')
    for i, level in enumerate(np.sort(levels)[::-1]):
        c = color[i,:]
        sub = shape_plot[shape_plot['level'] == level]
        plot = sub.boundary.plot(
            ax=ax,
            linewidth=1.5,
            edgecolor=c,
            label=level,
        )
        if (i == 0) & label:
            sub['coords'] = sub['geometry'].apply(lambda x: x.representative_point().coords[:])
            sub['coords'] = [coords[0] for coords in sub['coords']]
            for idx, row in sub.iterrows():
                plt.annotate(text=row['ADMIN%d' % int(level[-1])], xy=row['coords'],
                             horizontalalignment='center',color='dimgrey',fontsize=11)
        
    # plot.legend(fontsize=15,frameon=False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15,frameon=False)
    plt.tight_layout()
    # plt.show()
    plt.close(fig)
    country_code = shape_plot['COUNTRY'].unique()[0]
    fn_save = '../figures/%s_admin_shapes.png' % country_code
    fig.savefig(fn_save, bbox_inches='tight', facecolor='w',edgecolor='w')
    print('%s is saved.' % fn_save)
    return

def FDW_PD_ValidateFnidName(df, shape_used, shape_latest):
    '''Chekcing all FNID and Names between FDW data and FEWS NET Shapefiles for consistency.
    '''
    name_fdw = df[['fnid','admin_1','admin_2']].drop_duplicates()
    if 'ADMIN2' not in shape_used.columns: #admin 1 files may not have an admin 2 field, which would be entirely null
        name_shape = shape_used[['FNID','ADMIN1']].drop_duplicates()
    else:
        name_shape = shape_used[['FNID','ADMIN1','ADMIN2']].drop_duplicates()
        
    # Check all FNIDs exist in the shapefiles.
    fnid_not_in_shape = name_fdw[
        (name_fdw['fnid'].apply(lambda x: x[6] == 'A')) &
        (~name_fdw['fnid'].isin(name_shape['FNID']))
    ]
    # assert len(fnid_not_in_shape) == 0
    # Check all names are consistent
    name_merged = name_shape.merge(name_fdw, left_on='FNID', right_on='fnid')
    # - Admin level 1
    adm1 = name_merged['FNID'].apply(lambda x: x[6:8] == 'A1')
    name_merged_adm1 = name_merged.loc[adm1, ['FNID','ADMIN1', 'admin_1']]
    name_replace1 = name_merged_adm1[name_merged_adm1['ADMIN1'] != name_merged_adm1['admin_1']]
    for i, (FNID, ADMIN1, admin_1) in name_replace1.iterrows():
        print('%s:\t"%s" (FDW) is changed to "%s" (shapefile).' % (FNID, admin_1, ADMIN1))
        df.loc[df['fnid'] == FNID, 'admin_1'] = ADMIN1
    # - Admin level 2
    if 'ADMIN2' in shape_used.columns:
        adm2 = name_merged['FNID'].apply(lambda x: x[6:8] == 'A2')
        name_merged_adm2 = name_merged.loc[adm2, ['FNID','ADMIN2', 'admin_2']]
        name_replace2 = name_merged_adm2[name_merged_adm2['ADMIN2'] != name_merged_adm2['admin_2']]
        for i, (FNID, ADMIN2, admin_2) in name_replace2.iterrows():
            print('%s:\t"%s" (FDW) is changed to "%s" (shapefile).' % (FNID, admin_2, ADMIN2))
            df.loc[df['fnid'] == FNID, 'admin_2'] = ADMIN2
    # Define representative administrative names
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A1'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A1'), 'admin_1']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A2'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A2'), 'admin_2']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A3'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A3'), 'admin_3']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A4'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'A4'), 'admin_4']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R1'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R1'), 'admin_1']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R2'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R2'), 'admin_2']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R3'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R3'), 'admin_3']
    df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R4'), 'name'] = df.loc[df['fnid'].apply(lambda x: x[6:8] == 'R4'), 'admin_4']
    for gdf in [shape_used, shape_latest]:
        fdx = gdf.FNID.apply(lambda x: x[7] == '1')
        gdf.loc[fdx,'name'] = gdf.loc[fdx, 'ADMIN1']
        if ('ADMIN2' in gdf.columns):
            fdx = gdf.FNID.apply(lambda x: x[7] == '2')
            gdf.loc[fdx,'name'] = gdf.loc[fdx, 'ADMIN2']
    return df


def FDW_PD_Compiling(df, shape_used):
    # MultiIndex of all 'fnid', 'name', 'product', 'season_name', and 'season_date'
    # Use all FNIDs found in Shapefiles and FDW data
    fnid_name1 = df[['fnid','name']].drop_duplicates()
    fnid_name1.columns = ['FNID','name']
    fnid_name2 = shape_used[['FNID','name']].drop_duplicates()
    fnid_name = pd.concat([fnid_name1,fnid_name2],axis=0).drop_duplicates()
    fnid_name = fnid_name.sort_values('FNID').reset_index(drop=True)
    years = np.arange(df['harvest_year'].min(), df['harvest_year'].max()+1)
    products = df['product'].unique()
    season_variables = ['season_name','planting_month','harvest_month']
    seasons = df[season_variables].drop_duplicates()
    production_systems = df['crop_production_system'].unique()
    # Generate the MultiIndex columns
    mdx = [(f, n, p, sn, gm, hm, cps) for f, n in list(fnid_name.values.tolist()) 
           for p in products 
           for sn, gm, hm in list(seasons.values.tolist())
           for cps in production_systems
          ]
    mdx = pd.MultiIndex.from_tuples(mdx,names=['fnid','name','product',*season_variables,'crop_production_system'])
    # Create tables of "Area Harvested" and "Quantity Produced"
    # "harvest_year" is used as a standard year of the data
    area = df[df['indicator'] == 'Area Harvested']
    area = pd.pivot_table(area, index='harvest_year',
                          columns=['fnid','name','product',*season_variables,'crop_production_system'],
                          values='value',
                          aggfunc=lambda x: x.sum(min_count=1)
                         ).reindex(mdx,axis=1).reindex(years,axis=0)
    prod = df[df['indicator'] == 'Quantity Produced']
    prod = pd.pivot_table(prod, index='harvest_year',
                          columns=['fnid','name','product',*season_variables,'crop_production_system'],
                          values='value',
                          aggfunc=lambda x: x.sum(min_count=1)
                         ).reindex(mdx,axis=1).reindex(years,axis=0)
    assert area.shape == prod.shape
    return area, prod


def CreateLinkAdmin(shape_old, shape_new, old_on='ADMIN2', new_on='ADMIN2'):
    '''Algorithm to get links between administrative boundaries
    '''
    # Find link
    over = gpd.overlay(shape_old, shape_new, how='intersection', keep_geom_type=False)
    over = over.to_crs('EPSG:32629')
    over['area'] = over['geometry'].area / 10**6
    link = {}
    for i in shape_new.FNID:
        temp = over.loc[over['FNID_2'] == i, ['FNID_1', 'area']]
        link[i] = temp.iloc[temp.area.argmax()]['FNID_1']
    # Find link and name
    link_name = {}
    for l2, l1 in link.items():
        name2 = shape_new.loc[shape_new.FNID == l2, new_on].to_list()
        name1 = shape_old.loc[shape_old.FNID == l1, old_on].to_list()
        link_name['%s (%s)' % (l2, *name2)] = '%s (%s)' % (l1, *name1)
    # Find newly added units as an inverse form
    inv = invert_dicts(link_name)
    add = list(compress(list(inv.keys()), np.array([len(v) for k, v in inv.items()]) > 1))
    diff_inv = {k: inv[k] for k in add}
    return sort_dict(link), sort_dict(link_name), sort_dict(diff_inv)

def CountNumbCropCell(shape_org, fn_cropland):
    """
    This function counts the number of positive cropland cells per each feature of the shapefile.
    Donghoon Lee @ 2022.07.16
    """
    # Generate numeric ID for burning features into the raster
    shapefile = shape_org.to_crs('EPSG:4326').copy()
    shapefile.loc[:,'ID'] = np.arange(1,shapefile.shape[0]+1)
    shapefile['ID'] = shapefile['ID'].astype(rasterio.int16)
    # Clip the raster to the domain of the shapefile
    geojson = json.loads(shapefile['geometry'].to_json()) 
    shapes_clip = [i['geometry'] for i in geojson['features']]
    with rasterio.open(fn_cropland) as src:
        data, out_transform = mask(src, shapes_clip, crop=True)
        src_meta = src.meta
        src_meta.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": out_transform
        })    
    # Burning features into the raster
    shapes = ((geom,value) for geom, value in zip(shapefile.geometry, shapefile['ID']))
    burned = features.rasterize(shapes=shapes, 
                                fill=0, 
                                out_shape=[src_meta['height'], src_meta['width']], 
                                transform=src_meta['transform'])
    burned = burned.astype(rasterio.int16)
    for i in shapefile['ID'].unique():
        shapefile.loc[shapefile['ID'] == i, 'CropCell'] = sum(data.squeeze()[burned==i] > 0)
    return shapefile


def FDW_PD_CreateAdminLink(shape_old, shape_new, old_on, new_on, prod, crs):
    # Create intersection between shapefiles
    over = gpd.overlay(shape_old, shape_new, how='intersection', keep_geom_type=False)
    over = over.to_crs(crs)
    over['area'] = over['geometry'].area / 10**6
    over = CountNumbCropCell(over, '../data/cropmask/Hybrid_10042015v9.img')
    # Create Link between admin units
    # - Production-based Ratio (PBR) is applied when boundaries are matched with 10% of errors.
    # - Cropland-based Ratio (CBR) is applied when boundaries are not matched.
    # *** The identified links will be examined by manually.
    link = {}
    for fnid_old in shape_old.FNID:
        temp = over.loc[over['FNID_1'] == fnid_old, ['FNID_2', '%s_2' % new_on,'area']]
        temp = temp.rename(columns={'area': 'area_old'}).reset_index(drop=True)
        temp['area_new'] = 0
        for i, (FNID_2, _, area_old, _) in temp.iterrows():
            t = area_old/(shape_new.loc[shape_new['FNID'] == FNID_2, 'area'].values[0])*100
            temp.loc[temp['FNID_2'] == FNID_2, 'area_new'] = t
        temp['area_old'] = temp['area_old']/temp['area_old'].sum()*100
        temp = temp.loc[temp['area_new'] > 10]
        fnid_new = temp['FNID_2'].to_list()
        if (temp['area_old'].sum() > 90) & all(temp['area_new'] > 90):
            method = 'PBR'
            # if any fnids_new have no records, consider CBR
            no_record = prod[fnid_new].isna().all().groupby('fnid').all()
            if no_record.any():
                method = 'CBR'
                list_fnid_no_record = list(no_record.index[no_record])
                print("CBR is considered for '%s' as no record found in:" % (fnid_old), list_fnid_no_record)
        else:
            method = 'CBR'
        fnids = {}
        for fnid in fnid_new:
            fnids[fnid] = dict(
                name = shape_new.loc[shape_new['FNID'] == fnid, new_on].values[0],
            )
        link[fnid_old] = dict(
            name=shape_old.loc[shape_old['FNID'] == fnid_old, old_on].values[0],
            method=method, 
            fnids=sort_dict(fnids)
        )
    fnid_new_all = np.unique(np.concatenate([list(v['fnids'].keys()) for k,v in link.items()]))
    assert np.isin(shape_new['FNID'], fnid_new_all).all()
    return link, over


def FDW_PD_ValiateAdminLink(link):
    # Separate PBR and CBP
    link_PBR = sort_dict({k:v for k,v in link.items() if v['method'] == 'PBR'})
    link_CBR = sort_dict({k:v for k,v in link.items() if v['method'] == 'CBR'})
    # Validate no intersection between PBR and CBR
    fnids_PBR_old = list(link_PBR.keys())
    fnids_PBR_new = np.unique(np.concatenate([list(v['fnids'].keys()) for k, v in link_PBR.items()]))
    fnids_CBR_old = list(link_CBR.keys())
    if len(link_CBR) > 0:
        fnids_CBR_new = np.unique(np.concatenate([list(v['fnids'].keys()) for k, v in link_CBR.items()]))
    else:
        fnids_CBR_new = []
    assert np.isin(fnids_PBR_old, fnids_CBR_old).any() == False
    assert np.isin(fnids_PBR_new, fnids_CBR_new).any() == False
    # Inverse link_CBR
    link_CBR_sim = dict()
    for k,v in link_CBR.items():
        link_CBR_sim[k] = list(v['fnids'].keys())
    link_CBR_inv = invert_dict(link_CBR_sim)
    return link_PBR, link_CBR, link_CBR_sim, link_CBR_inv


def FDW_PD_RatioAdminLink(link, prod, over, mdx_pss):
    '''Calculate (a) Production-based Ratio (PBR) and (b) Cropland-based Ratio (CBR)
    '''
    mdx_pss_name = ['product','season_name','planting_month','harvest_month','crop_production_system']
    # Validation
    link_PBR, link_CBR, link_CBR_sim, link_CBR_inv = FDW_PD_ValiateAdminLink(link)
    # Calculate specific ratios between AdminLink
    fnids_in_link = np.unique(np.concatenate([list(v['fnids'].keys()) for k,v in link.items()]))
    fnids_in_data = prod.columns.get_level_values(0).unique()
    fnids_target = fnids_in_link[np.isin(fnids_in_link, fnids_in_data)]
    link_ratio = dict()
    for fnid in sorted(fnids_target):
        link_ratio[fnid] = pd.DataFrame(index=mdx_pss, dtype=np.float32)
    # (a) Production-based Ratio (PBR)
    for k,v in link_PBR.items():
        fnid_old, fnids_new = k, list(v['fnids'].keys())
        # - Ignore districts not in the data
        fnids_new = [fnid for fnid in fnids_new if fnid in fnids_in_data]
        # - Ignore districts if no data exist
        fnids_new = [fnid for fnid in fnids_new if np.sum(prod[fnid].notna()).sum() > 0]
        if len(fnids_new) == 0: continue
        prod_mean = prod.loc[:, pd.IndexSlice[fnids_new]].mean()
        # prod_mean[prod_mean.isna()] = 0
        ratio = prod_mean.divide(prod_mean.groupby(mdx_pss_name).sum())
        # ratio[ratio.isna()] = 0
        ratio = ratio.rename('ratio').reset_index().pivot_table(index=mdx_pss_name, columns=['fnid'],values='ratio')
        # Data exists in "fnid_old" but not in "fnid_new" 
        exist_old = prod[fnid_old].columns[prod[fnid_old].sum() > 0].droplevel(0)
        exist_new = ratio.index
        no_new_record = exist_old[~exist_old.isin(exist_new)]
        ratio = ratio.reindex(ratio.index.append(no_new_record))
        ratio.loc[no_new_record, fnids_new] = 1/len(fnids_new)
        for fnid_new in fnids_new:
            # Insert ratios
            link_ratio[fnid_new][fnid_old] = ratio[fnid_new]
    if len(link_CBR) > 0:
        # (b) Cropland-based Ratio (CBR)
        # Pre-calculate total cropland area (area_old_all) to avoid sum(ratios) < 1
        table_area_old_all = dict()
        for fnid_old, fnids_new in link_CBR_sim.items():
            table_area_old_all[fnid_old] = over.loc[(over['FNID_1'] == fnid_old) & (over['FNID_2'].isin(fnids_new)), 'CropCell'].sum()
            table_area_old_all = pd.Series(table_area_old_all)
        # Scale time-series with ratios of cropland areas
        for fnid_new, fnids_old in link_CBR_inv.items():
            area_old_all = table_area_old_all[fnids_old]
            area_old_part = over[(over['FNID_2'] == fnid_new)&(over['FNID_1'].isin(fnids_old))].groupby('FNID_1')[['area','CropCell']].sum()
            ratio = area_old_part['CropCell']/area_old_all
            for fnid_old in fnids_old:
                link_ratio[fnid_new][fnid_old] = ratio[fnid_old]
    return link_ratio


def FDW_PD_ConnectAdminLink(link_ratio, area, prod, validation=True):
    # Connect linked production data
    area_new = []
    prod_new = []
    for fnid_new in link_ratio.keys():
        name_new = area[fnid_new].columns[0][0]
        ratio = link_ratio[fnid_new]
        #drop duplicates in case of merging countries that split with differing subsequent unit sets
        # For example, after SD and SS spilt, SS only has 2011, while SD has 2011, 2012, 2013 etc. This results in duplicates if 2011 SS
        # is appended to 2011, 2012, and 2013 SD to create continuous boundaries across new country boundaries
        ratio = ratio.iloc[:,~(ratio.columns.duplicated())]
        area_scaled = []
        prod_scaled = []
        for fnid in ratio.columns:
            area_scaled.append(area[fnid].multiply(ratio[fnid]).droplevel(0,axis=1))
            prod_scaled.append(prod[fnid].multiply(ratio[fnid]).droplevel(0,axis=1))
        # Merge all scaled data
        area_merged = reduce(lambda a, b: a.add(b, fill_value=0), area_scaled)
        prod_merged = reduce(lambda a, b: a.add(b, fill_value=0), prod_scaled)
        # Restore columns (FNID and Name)
        area_merged = pd.concat({fnid_new:pd.concat({name_new:area_merged},names=['name'],axis=1)},names=['fnid'],axis=1)
        prod_merged = pd.concat({fnid_new:pd.concat({name_new:prod_merged},names=['name'],axis=1)},names=['fnid'],axis=1)
        area_new.append(area_merged)
        prod_new.append(prod_merged)
    area_new = pd.concat(area_new, axis=1)
    prod_new = pd.concat(prod_new, axis=1)
    if validation:
        # assert np.isclose(area_new.sum(1), area.sum(1)).all() == True
        # assert np.isclose(prod_new.sum(1), prod.sum(1)).all() == True
        assert sum(abs((area_new.sum(1) - area.sum(1))/(area.sum(1) + 0.01)) > 0.01) == 0  # less than 1% difference is allowed
        assert sum(abs((prod_new.sum(1) - prod.sum(1))/(prod.sum(1) + 0.01)) > 0.01) == 0  # less than 1% difference is allowed
    return area_new, prod_new


def FDW_PD_CaliSeasonYear(stack, esc, link_ratio=None):
    # Trim all the text in the tables
    stack[['country','season_name','product','crop_production_system']] = stack[['country','season_name','product','crop_production_system']].applymap(lambda x: x.strip())
    esc[['country','season_name']] = esc[['country','season_name']].applymap(lambda x: x.strip())
    # Check all rows of cspc_table_stack are in cspc_table_ecc
    cs_table_stack = stack[['country','season_name']].drop_duplicates().sort_values(by=['country','season_name']).reset_index(drop=True)
    cs_table_ecs = esc[['country','season_name']].drop_duplicates().sort_values(by=['country','season_name']).reset_index(drop=True)
    # Use merge to find matching rows, with an indicator to show the match status
    result = pd.merge(cs_table_stack, cs_table_ecs, how='left', on=['country','season_name'], indicator=True)
    try: 
        assert result['_merge'].eq('both').all()
        print('All [country, season_name] are in the external season calendar.')
    except:
        print('Below data are not in external season calendar:')
        print(result[~result['_merge'].eq('both')].to_string())
        # Warning message
        raise ValueError('Some data are not in the external season calendar. Please check the data.')
    # Calibrate "stack"
    for (c, s, pm, hm, py, hy) in esc.values:
        query_str = f'country == "{c}" and season_name == "{s}"'
        stack.loc[stack.query(query_str).index, 'planting_month'] = pm
        stack.loc[stack.query(query_str).index, 'planting_year'] += py
        stack.loc[stack.query(query_str).index, 'harvest_month'] = hm
        stack.loc[stack.query(query_str).index, 'harvest_year'] += hy
    if link_ratio is not None:
        # Calibrate "link_ratio"
        esc_season = esc[['season_name','planting_month','harvest_month']].drop_duplicates()
        for fnid, ratio in link_ratio.items():
            mdx = ratio.index.to_frame().reset_index(drop=True)
            for (s, pm, hm) in esc_season.values:
                query_str = f'season_name == "{s}"'
                mdx.loc[mdx.query(query_str).index, 'planting_month'] = pm
                mdx.loc[mdx.query(query_str).index, 'harvest_month'] = hm
            ratio.index = pd.MultiIndex.from_frame(mdx)
            link_ratio[fnid] = ratio
    return stack, link_ratio if link_ratio is not None else stack


def FDW_PD_GrainTypeAgg(list_table, product_category):
    # Change product name to grain type (e.g., "Maize (Corn)" -> "Maize")
    print("- Aggregation of grain types ------------------ #")
    list_product = sorted(list_table[0].columns.levels[2].unique())
    print("%d crops: %s" % (len(list_product), ", ".join(sorted(list_product))))
    for i, temp in enumerate(list_table):
        # Tailored MultiIndex
        mdx = temp.columns.to_frame().reset_index(drop=True)
        mdx['product'].replace(product_category,inplace=True)
        mdx = mdx.drop_duplicates()
        # Change product types and aggregate the same products
        temp = temp.T.stack().rename('value').reset_index()
        temp['product'] = temp['product'].replace(product_category)
        temp = temp.groupby(list(temp.columns[:-1])).sum(min_count=1).reset_index()
        temp = temp.pivot_table(index=temp.columns[-2],columns=temp.columns[:-2].tolist(),values='value')
        temp = temp.reindex(columns=pd.MultiIndex.from_frame(mdx))
        list_table[i] = temp
    list_product = sorted(list_table[0].columns.levels[2].unique())
    print("%d crops: %s" % (len(list_product), ", ".join(sorted(list_product))))
    print('')
    return list_table


def FDW_PD_MergeCropProductionSystem(area_new, prod_new, cps_remove, cps_final):
    # Area
    area_new = area_new.drop(cps_remove, level=6, axis=1)
    area_new = area_new.sum(level=[0,1,2,3,4,5], axis=1, min_count=1)
    col_new = area_new.columns.to_frame().reset_index(drop=True)
    col_new['crop_production_system'] = cps_final
    area_new.columns = pd.MultiIndex.from_frame(col_new)
    # Production
    prod_new = prod_new.drop(cps_remove, level=6, axis=1)
    prod_new = prod_new.sum(level=[0,1,2,3,4,5], axis=1, min_count=1)
    col_new = prod_new.columns.to_frame().reset_index(drop=True)
    col_new['crop_production_system'] = cps_final
    prod_new.columns = pd.MultiIndex.from_frame(col_new)
    return area_new, prod_new



def miss_pct(df,admin_level,country,country_iso,fnidMap):
    if (admin_level != 2) & (admin_level != 3):
        return
    if 'none' in df[df['country'] == country]['admin_2'].unique():
        return
    if country == 'Tanzania':
        df['country'].replace('Tanzania, United Republic of', 'Tanzania', inplace=True)
    admMissPct = pd.DataFrame(columns=['country', 'country_code', 'admin_1', 'admin_2', 'name', 'product', 'planting_year',
                 'planting_month', 'harvest_year', 'harvest_month', 'season_name', 'crop_production_system',
                 'missing_pct'])
    FillDF=pd.DataFrame(columns=df.columns[:])
    for crop in df['product'].unique():
        regionMissInd=pd.DataFrame()
        couDF = df[(df['country']==country)&(df['product']==crop)]
        if len(couDF) == 0:
            continue
        cropSys = couDF.crop_production_system.unique()
        for crop_sys in cropSys:
            for season_name in couDF[couDF['crop_production_system']==crop_sys].season_name.unique():
                csDF = couDF[(couDF.season_name == season_name) & (couDF.crop_production_system == crop_sys)]
                csDF =  csDF.drop_duplicates()
                names = csDF.name.unique()
                calDF = pd.DataFrame(columns=df.columns[:])
                fillDF = pd.DataFrame(columns=df.columns[:])
                #  Calculate the value using other two version
                missingYrName = pd.DataFrame(columns=['name', 'year', 'indicator'])
                for name in names:
                    rDF = csDF[csDF.name == name]
                    for iYr in rDF.harvest_year.unique():
                        payDF = rDF[rDF.harvest_year == iYr]
                        if len(payDF) == 2:
                            fill = pd.DataFrame(payDF.copy().iloc[0:1, :])
                            if 'Quantity Produced' not in payDF.indicator.values:
                                fill.loc[:, 'indicator'] = 'Quantity Produced'
                                fill.loc[:, 'value'] = payDF[payDF.indicator == 'Area Harvested'].value.values[0] * \
                                                       payDF[payDF.indicator == 'Yield'].value.values[0]
                            if 'Area Harvested' not in payDF.indicator.values:
                                fill.loc[:, 'indicator'] = 'Area Harvested'
                                fill.loc[:, 'value'] = payDF[payDF.indicator == 'Quantity Produced'].value.values[0] / \
                                                       payDF[payDF.indicator == 'Yield'].value.values[0]
                            if 'Yield' not in payDF.indicator.values:
                                fill.loc[:, 'indicator'] = 'yield'
                                fill.loc[:, 'value'] = payDF[payDF.indicator == 'Quantity Produced'].value.values[0] / \
                                                       payDF[payDF.indicator == 'Area Harvested'].value.values[0]
                            fill.loc[:, 'fill_status'] = 'calculated'
                            calDF = pd.concat([fill, calDF])
                        if (len(payDF) == 1):
                            missingYrName.loc[len(missingYrName)] = [name, iYr, payDF.indicator.values[0]]
                            # print('*********************************')
                csDF = pd.concat([csDF, calDF])
                # filled values ,expected valuesa
                for name in names:
                    for iInd in ['Quantity Produced', 'Area Harvested', 'Yield']:
                        # limit to admin unit
                        iDF = csDF[(csDF.name == name) & (csDF.indicator == iInd)]
                        if len(iDF) == 0:
                            continue
                        dfYrs = iDF.planting_year.unique()
                        yrs = np.arange(dfYrs.min(), dfYrs.max(), 1)
                        # find missing years
                        missYrs = yrs[~np.isin(yrs, dfYrs)]
                        # identify potential offsets for harvest year and planting year
                        hYrOffset = iDF.loc[:, 'harvest_year'].values[0] - iDF.loc[:, 'planting_year'].values[0]

                        # add in 0s for production, harvested area, and yield
                        if len(missYrs) > len(iDF):
                            fill = pd.DataFrame(csDF.copy().iloc[0:missYrs.size, :])
                            fill.loc[:, ['admin_1']] = iDF.admin_1.values[0]
                            fill.loc[:, ['admin_2']] = iDF.admin_2.values[0]
                            fill.loc[:, ['name']] = iDF.name.values[0]
                            fill.loc[:, ['planting_year']] = missYrs
                            fill.loc[:, ['harvest_year']] = missYrs + hYrOffset
                            fill.loc[:, 'value'] = np.average(iDF.value.values)
                            fill.loc[:, 'fill_status'] = 'filled'
                            fill.loc[:, 'indicator'] = iInd
                            # add the missing value where it goes
                            iDF = pd.concat([iDF, fill])
                            iDF = iDF.sort_values('harvest_year')
                            # create a dataframe with the smoothed values
                            iDFsmooth = pd.DataFrame(iDF.copy().iloc[:])
                            iDFsmooth.loc[:, 'indicator'] = str(iInd + '_smooth')

                            # concat both the data with added 0s and the smoothed data onto the master filled DF
                            fillDF = pd.concat([fillDF, iDF, iDFsmooth])
                        else:
                            fill = pd.DataFrame(iDF.copy().iloc[0:missYrs.size, :])
                            fill.loc[:, ['planting_year']] = missYrs
                            fill.loc[:, ['harvest_year']] = missYrs + hYrOffset
                            fill.loc[:, 'value'] = 0
                            fill.loc[:, 'fill_status'] = 'filled'
                            fill.loc[:, 'indicator'] = iInd

                            # add the missing value where it goes
                            iDF = pd.concat([iDF, fill])
                            iDF = iDF.sort_values('harvest_year')

                            # calculate 5-year running mean of production
                            # To account for a missing values in locations we can drop and re-weight the remaining values by counting non-zero values
                            count = np.copy(iDF.value)
                            count[iDF.fill_status == 'data'] = 1
                            count[iDF.fill_status == 'calculated'] = 1
                            count[iDF.fill_status == 'filled'] = 0

                            gauExp = ndimage.gaussian_filter1d(iDF.value.values, 3) / ndimage.gaussian_filter1d(count,
                                                                                                                3)

                            # create a dataframe with the smoothed values
                            iDFsmooth = pd.DataFrame(iDF.copy().iloc[:])
                            iDFsmooth.loc[:, 'indicator'] = str(iInd + '_smooth')
                            iDFsmooth.loc[:, 'value'] = gauExp

                            # concat both the data with added 0s and the smoothed data onto the master filled DF
                            fillDF = pd.concat([fillDF, iDF, iDFsmooth])

                for i in range(len(missingYrName)):
                    missYr = missingYrName.year[i]
                    missName = missingYrName.name[i]
                    missIndicator = missingYrName.indicator[i]
                    missDF = fillDF[(fillDF.name == missName) & (fillDF.harvest_year == missYr)]
                    missDF['missing_status'] = 'missing'
                    if 'yield' == missIndicator:
                        if ('Quantity Produced_smooth' in missDF.indicator.values) & (
                                'Area Harvested_smooth' in missDF.indicator.values):
                            missDF.loc[missDF.indicator == 'Quantity Produced', 'value'] = missDF.loc[
                                                                                        missDF.indicator == 'Area Harvested_smooth', 'value'] * \
                                                                                    missDF.loc[
                                                                                        missDF.indicator == 'Yield', 'value']
                        if ('Quantity Produced_smooth' in missDF.indicator.values) & (
                                'Area Harvested_smooth' not in missDF.indicator.values):
                            tempAreaDF = pd.DataFrame(missDF.copy().loc[(missDF.harvest_year == missYr) & (
                                        missDF.indicator == str('Quantity Produced'))])
                            tempAreaDF.loc[:, 'indicator'] = 'Area Harvested'
                            tempAreaDF.loc[tempAreaDF.indicator == 'Area Harvested', 'value'] = missDF.loc[
                                                                                          missDF.indicator == 'production_smooth', 'value'] / \
                                                                                      missDF.loc[
                                                                                          missDF.indicator == 'Yield', 'value']
                            missDF = pd.concat(missDF, tempAreaDF)
                        if ('Quantity Produced_smooth' not in missDF.indicator.values) & (
                                'Area Harvested_smooth' in missDF.indicator.values):
                            tempProdDF = pd.DataFrame(missDF.copy().loc[(missDF.harvest_year == missYr) & (
                                        missDF.fillstatus == str('filled'))])
                            tempProdDF = tempProdDF.replace({'Area Harvested': 'Quantity Produced', 'Area Harvested_smooth': 'Quantity Produced_smooth'})
                            tempProdDF.loc[tempProdDF.indicator == 'Quantity Produced_smooth', 'value'] = missDF.loc[
                                                                                                       missDF.indicator == 'Yield_smooth', 'value'] * \
                                                                                                   missDF.loc[
                                                                                                       missDF.indicator == 'Area Harvested_smooth', 'value']
                            tempProdDF.loc[tempProdDF.indicator == 'Quantity Produced', 'value'] = missDF.loc[
                                                                                                missDF.indicator == 'Yield', 'value'] * \
                                                                                            missDF.loc[
                                                                                                missDF.indicator == 'Area Harvested_smooth', 'value']
                            missDF = pd.concat(missDF, tempProdDF)
                    if len(csDF)>1:
                        fillDF[(fillDF.name == missName) & (fillDF.harvest_year == missYr)] = missDF[
                            (missDF.name == missName) & (missDF.harvest_year == missYr)]

                missPartCS = fillDF[fillDF.missing_status == 'missing']
                regionMissInd = pd.concat([missPartCS, regionMissInd])

                # calculate the percent of subnational production at each time step based on the smoothed data and use it to fill the data
                if admin_level==2:               
                    cntPrd = fillDF.loc[
                        fillDF.indicator == 'Quantity Produced_smooth', ['country','country_code','admin_1','product', 'season_name','planting_year','planting_month', 'harvest_year','harvest_month','crop_production_system','value',]].groupby(
                        ['country','country_code','admin_1','product', 'season_name','planting_year','planting_month', 'harvest_year','harvest_month','crop_production_system']).sum()
                    cntPrd.reset_index(inplace=True)
                    cntPrd['admin_2']='None'
                    cntPrd['name']=cntPrd['admin_1']
                    for iYr in fillDF.harvest_year.unique():
                        pctPrd = pd.DataFrame(fillDF.copy().loc[(fillDF.harvest_year == iYr) & (
                                    fillDF.indicator == str('Quantity Produced_smooth'))])
                        if len(pctPrd) == 0:
                            continue
                        for admin1 in pctPrd.admin_1.unique():
                            pctNamePrd = pctPrd[pctPrd.admin_1 == admin1]
                            pctNamePrd.loc[:, 'value'] = pctNamePrd.value / cntPrd.loc[(cntPrd.admin_1==admin1)&(cntPrd.harvest_year==iYr),'value'].values
                            pctNamePrd['indicator'] = 'production_fraction'
                            fillDF = pd.concat([fillDF, pctNamePrd])
    
                    # indentify when fill_status is 'filled' and indicator is 'production_fraction' or fill_status is 'data' and indicator is 'production_fraction' and missing_status is 'missing'
                    rPctPrd0 = pd.DataFrame(fillDF.copy().loc[(fillDF.fill_status == str('filled')) & (
                                fillDF.indicator == str('production_fraction'))])
                    rPctPrd1 = pd.DataFrame(fillDF.copy().loc[(fillDF.missing_status == str('missing')) & (
                                fillDF.indicator == str('production_fraction')) & (fillDF.fill_status == str('data'))])
                    rPctPrd = pd.concat([rPctPrd0, rPctPrd1])
                    missPct = pd.DataFrame(columns=['country', 'country_code', 'admin_1', 'admin_2', 'name', 'product', 'planting_year',
                     'planting_month', 'harvest_year', 'harvest_month', 'season_name', 'crop_production_system',
                     'missing_pct'])
                    for iYr in rPctPrd.harvest_year.unique():
                        pctPrd = rPctPrd.copy().loc[rPctPrd.harvest_year == iYr]
                        for admin1 in pctPrd.admin_1.unique():
                            admin1Pct=pctPrd[(pctPrd['admin_1'] == admin1)]
                            # adm1Fnid=gdf[gdf.ADMIN1==admin1].FNID.values[0]
                            for name in admin1Pct.name.unique():
                                namePct=pctPrd[(pctPrd['admin_1'] == admin1) & (pctPrd.name == name)]
                                sumPct = pctPrd[(pctPrd['admin_1'] == admin1) & (pctPrd.name == name)]['value'].sum()
    
                                missPct.loc[len(missPct)] = [country,namePct['country_code'].values[0], admin1,namePct.admin_2.values[0], name,crop,namePct.planting_year.values[0],namePct.planting_month.values[0], iYr, namePct.harvest_month.values[0],season_name, crop_sys,
                                                             sumPct]
                    admMissPct = pd.concat([missPct, admMissPct])
                    FillDF=pd.concat([FillDF,fillDF])
                if admin_level == 3:
                    cntPrd = fillDF.loc[
                        fillDF.indicator == 'Quantity Produced_smooth', ['country', 'country_code', 'admin_1','admin_2', 'product',
                                                                  'season_name', 'planting_year', 'planting_month',
                                                                  'harvest_year', 'harvest_month',
                                                                  'crop_production_system',
                                                                  'value', ]].groupby(
                        ['country', 'country_code', 'admin_1','admin_2', 'product', 'season_name', 'planting_year',
                         'planting_month', 'harvest_year', 'harvest_month', 'crop_production_system',
                         ]).sum()
                    cntPrd.reset_index(inplace=True)
                    cntPrd['admin_3'] = 'None'
                    cntPrd['name'] = cntPrd['admin_2']
                    for iYr in fillDF.harvest_year.unique():
                        pctPrd = pd.DataFrame(fillDF.copy().loc[(fillDF.harvest_year == iYr) & (
                                fillDF.indicator == str('Quantity Produced_smooth'))])
                        if len(pctPrd) == 0:
                            continue
                        for admin2 in pctPrd.admin_2.unique():
                            pctNamePrd = pctPrd[pctPrd.admin_2 == admin2]
                            pctNamePrd.loc[:, 'value'] = pctNamePrd.value / cntPrd.loc[
                                (cntPrd.admin_2 == admin2) & (cntPrd.harvest_year == iYr), 'value'].values
                            pctNamePrd['indicator'] = 'production_fraction'
                            fillDF = pd.concat([fillDF, pctNamePrd])

                    # indentify when fill_status is 'filled' and indicator is 'production_fraction' or fill_status is 'data' and indicator is 'production_fraction' and missing_status is 'missing'
                    rPctPrd0 = pd.DataFrame(fillDF.copy().loc[(fillDF.fill_status == str('filled')) & (
                            fillDF.indicator == str('production_fraction'))])
                    rPctPrd1 = pd.DataFrame(fillDF.copy().loc[(fillDF.missing_status == str('missing')) & (
                            fillDF.indicator == str('production_fraction')) & (fillDF.fill_status == str('data'))])
                    rPctPrd = pd.concat([rPctPrd0, rPctPrd1])
                    missPct = pd.DataFrame(
                        columns=['country', 'country_code', 'admin_1', 'admin_2','admin_3', 'name', 'product', 'planting_year',
                                 'planting_month', 'harvest_year', 'harvest_month', 'season_name',
                                 'crop_production_system',
                                 'missing_pct', ])
                    for iYr in rPctPrd.harvest_year.unique():
                        pctPrd = rPctPrd.copy().loc[rPctPrd.harvest_year == iYr]
                        for admin2 in pctPrd.admin_2.unique():
                            admin2Pct = pctPrd[(pctPrd['admin_2'] == admin2)]
                            # adm1Fnid=gdf[gdf.ADMIN1==admin1].FNID.values[0]
                            for name in admin2Pct.name.unique():
                                namePct = pctPrd[(pctPrd['admin_2'] == admin2) & (pctPrd.name == name)]
                                sumPct = pctPrd[(pctPrd['admin_2'] == admin2) & (pctPrd.name == name)]['value'].sum()

                                missPct.loc[len(missPct)] = [country, namePct['country_code'].values[0], namePct.admin_1.values[0],
                                                             admin2,namePct.admin_3.values[0] ,name, crop,
                                                             namePct.planting_year.values[0],
                                                             namePct.planting_month.values[0], iYr,
                                                             namePct.harvest_month.values[0], season_name, crop_sys,
                                                             sumPct]
                    admMissPct = pd.concat([missPct, admMissPct])
                    FillDF = pd.concat([FillDF, fillDF])
    if admin_level == 2:
        admMissPct['fnid'] = admMissPct['admin_1']
        admMissPct['fnid'] = admMissPct['fnid'].map(fnidMap)
    if admin_level == 3:
        admMissPct['fnid'] = admMissPct['admin_2']
        admMissPct['fnid'] = admMissPct['fnid'].map(fnidMap)
    return admMissPct,FillDF

def agg_admin1(df,admin_level,country,country_iso,fnidMap):
    if (admin_level != 2) & (admin_level != 3):
        return
    if 'none' in df[df['country'] == country]['admin_2'].unique():
        return

    aggYield1 = pd.DataFrame(
        columns=['fnid', 'country', 'country_code', 'admin_1', 'admin_2','admin_3','name', 'product', 'season_name','planting_year',
                 'planting_month', 'harvest_year', 'harvest_month',  'crop_production_system',
                 'indicator','value'])
    aggActualAdmin1 = pd.DataFrame(
        columns=['fnid','country','country_code', 'admin_1','admin_2', 'name', 'product','season_name','planting_year','planting_month', 'harvest_year', 'harvest_month','crop_production_system', 'indicator',
                 'value'])
    admMissPct,fillDF=miss_pct(df, admin_level, country,country_iso,fnidMap)

    for crop in fillDF['product'].unique():

        couDF = fillDF[(fillDF['country'] == country) & (fillDF['product'] == crop)]
        couMiss= admMissPct[(admMissPct['country'] == country) & (admMissPct['product'] == crop)]
        if len(couDF) == 0:
            continue
        cropSys = couDF.crop_production_system.unique()
        for crop_sys in cropSys:
            couMissSys=couMiss[couMiss['crop_production_system']==crop_sys]
            for season_name in couDF[couDF['crop_production_system'] == crop_sys].season_name.unique():
                csDF = couDF[(couDF.season_name == season_name) & (couDF.crop_production_system == crop_sys)]
                missPct=couMissSys[couMissSys['season_name']==season_name]
                names = csDF.name.unique()
                csDF.reset_index(inplace=True)
                csDF.drop(columns='index', axis=1, inplace=True)
                markDFMissing = pd.DataFrame(columns=df.columns[:])
                for name in names:
                        for year in missPct[missPct.name == name].harvest_year:
                            iMissingPct = missPct[(missPct.name == name) & (missPct.harvest_year == year)]
                            if iMissingPct['missing_pct'].values[0] > 0.5:
                                markDF = csDF[(csDF.name == name) & (csDF.harvest_year == year)]
                                markDF['missing_status'] = 'missing'
                                csDF[(csDF.name == name) & (csDF.harvest_year == year)] = markDF[
                                    (markDF.name == name) & (markDF.harvest_year == year)]
                                markDFMissing = pd.concat([markDFMissing, markDF])
                csDF = pd.concat([csDF, markDFMissing])
                aggDF = csDF.drop_duplicates(keep=False)
                aggDF = aggDF[aggDF['fill_status'] == 'data']
                aggDF.reset_index(inplace=True)
                aggDF.drop(columns='index', axis=1, inplace=True)

                for name in aggDF.name.unique():
                        iDF = aggDF[(aggDF.name == name)]
                        iDFProd = iDF[(iDF.name == name) & (iDF.indicator == 'Quantity Produced')]
                        iDFArea = iDF[(iDF.name == name) & (iDF.indicator == 'Area Harvested')]

                        admin1 = iDF.admin_1.unique()[0]
                        admin2 = iDF.admin_2.unique()[0]
                        if admin_level==2:
                            if len(iDFArea) > len(iDFProd):
                                for year in iDFArea.harvest_year:
                                    if len(iDFProd[iDFProd.harvest_year == year]) == 0:
                                        aggDF.drop(aggDF[(aggDF.name == name) & (aggDF.harvest_year == year) & (
                                                aggDF.indicator == 'area') & (aggDF.admin_1 == admin1)].index,
                                                    inplace=True)
                            elif len(iDFArea) < len(iDFProd):
                                for year in iDFProd.harvest_year:
                                    if len(iDFArea[iDFArea.harvest_year == year]) == 0:
                                        aggDF.drop(aggDF[(aggDF.name == name) & (aggDF.harvest_year == year) & (
                                                aggDF.indicator == 'production') & (aggDF.admin_1 == admin1)].index,
                                                    inplace=True)
                        if admin_level==3:
                            if len(iDFArea) > len(iDFProd):
                                for year in iDFArea.harvest_year:
                                    if len(iDFProd[iDFProd.harvest_year == year]) == 0:
                                        aggDF.drop(aggDF[(aggDF.name == name) & (aggDF.harvest_year == year) & (
                                                aggDF.indicator == 'area') & (aggDF.admin_2 == admin2)].index,
                                                   inplace=True)
                            elif len(iDFArea) < len(iDFProd):
                                for year in iDFProd.harvest_year:
                                    if len(iDFArea[iDFArea.harvest_year == year]) == 0:
                                        aggDF.drop(aggDF[(aggDF.name == name) & (aggDF.harvest_year == year) & (
                                                aggDF.indicator == 'production') & (aggDF.admin_2 == admin2)].index,
                                                   inplace=True)
                if admin_level==2:
                    aggProd = aggDF.loc[
                            aggDF.indicator == 'Quantity Produced', ['country','country_code', 'admin_1','product','season_name','planting_year','planting_month','harvest_year','harvest_month',
                                                            'crop_production_system', 'value']].groupby(
                            ['country','country_code', 'admin_1','product','season_name','planting_year','planting_month','harvest_year','harvest_month',
                                                            'crop_production_system' ]).sum()
                    aggArea = aggDF.loc[
                            aggDF.indicator == 'Area Harvested', ['country','country_code', 'admin_1','product','season_name','planting_year','planting_month','harvest_year','harvest_month',
                                                            'crop_production_system', 'value']].groupby(
                            ['country','country_code', 'admin_1','product','season_name','planting_year','planting_month','harvest_year','harvest_month',
                                                            'crop_production_system']).sum()
                if admin_level == 3:
                    aggProd = aggDF.loc[
                        aggDF.indicator == 'Quantity Produced', ['country', 'country_code', 'admin_1','admin_2', 'product',
                                                          'season_name', 'planting_year', 'planting_month',
                                                          'harvest_year', 'harvest_month',
                                                          'crop_production_system', 'value']].groupby(
                        ['country', 'country_code', 'admin_1','admin_2', 'product', 'season_name', 'planting_year',
                         'planting_month', 'harvest_year', 'harvest_month',
                         'crop_production_system',]).sum()
                    aggArea = aggDF.loc[
                        aggDF.indicator == 'Area Harvested', ['country', 'country_code', 'admin_1','admin_2', 'product', 'season_name',
                                                    'planting_year', 'planting_month', 'harvest_year', 'harvest_month',
                                                    'crop_production_system', 'value']].groupby(
                        ['country', 'country_code', 'admin_1','admin_2', 'product', 'season_name', 'planting_year',
                         'planting_month', 'harvest_year', 'harvest_month',
                         'crop_production_system']).sum()
                aggProd.reset_index(inplace=True)
                aggArea.reset_index(inplace=True)
                if len(aggProd) != len(aggArea):
                    if admin_level==2:
                        for admin1 in aggArea.admin_1.unique():
                            iDFArea = aggArea[aggArea.admin_1 == admin1]
                            iDFProd = aggProd[aggProd.admin_1 == admin1]
                            if len(iDFArea) > len(iDFProd):
                                for year in iDFArea.harvest_year:
                                    if len(iDFProd[iDFProd.harvest_year == year]) == 0:
                                        aggArea.drop(
                                            aggArea[(aggArea.admin_1 == admin1) & (aggArea.harvest_year == year)].index,
                                            inplace=True)
                            elif len(iDFArea) < len(iDFProd):
                                for year in iDFProd.harvest_year:
                                    if len(iDFArea[iDFArea.harvest_year == year]) == 0:
                                        aggProd.drop(
                                            aggProd[(aggProd.admin_1 == admin1) & (aggProd.harvest_year == year)].index,
                                            inplace=True)
                    if admin_level==3:
                        for admin2 in aggArea.admin_2.unique():
                            iDFArea = aggArea[aggArea.admin_2 == admin2]
                            iDFProd = aggProd[aggProd.admin_2 == admin2]
                            if len(iDFArea) > len(iDFProd):
                                for year in iDFArea.harvest_year:
                                    if len(iDFProd[iDFProd.harvest_year == year]) == 0:
                                        aggArea.drop(
                                            aggArea[(aggArea.admin_2 == admin2) & (aggArea.harvest_year == year)].index,
                                            inplace=True)
                            elif len(iDFArea) < len(iDFProd):
                                for year in iDFProd.harvest_year:
                                    if len(iDFArea[iDFArea.harvest_year == year]) == 0:
                                        aggProd.drop(
                                            aggProd[(aggProd.admin_2 == admin2) & (aggProd.harvest_year == year)].index,
                                            inplace=True)
                aggProd.reset_index(inplace=True)
                aggArea.reset_index(inplace=True)
                tempProd = aggProd.copy()
                aggYld = tempProd
                aggYld['value'] = tempProd['value'] / aggArea['value']
                aggYield1 = pd.concat([aggYield1, aggYld])
                aggProd['indicator'] = 'Quantity Produced'
                aggArea['indicator'] = 'Area Harvested'
                aggYld['indicator'] = 'Yield'
                # aggYld.drop(columns='index', axis=1, inplace=True)
                aggAdmin = pd.concat([aggProd, aggArea, aggYld])
                aggActualAdmin1 = pd.concat([aggActualAdmin1, aggAdmin])
    aggActualAdmin1.drop(columns='index', axis=1, inplace=True)
    aggYield1.drop(columns='index', axis=1, inplace=True)
    aggYield1['season_name'].replace('Long/Dry', 'Long(Dry)', inplace=True)
    if admin_level == 2:
        aggActualAdmin1['admin_2']='None'
        aggActualAdmin1['name'] = aggActualAdmin1['admin_1']
        aggYield1['admin_2']='None'
        aggYield1['name'] = aggYield1['admin_1']
        aggYield1['indicator'] ='Yield'
        aggYield1['fnid'] = aggYield1['admin_1']
        aggYield1['fnid'] = aggYield1['fnid'].map(fnidMap)
        aggActualAdmin1['fnid'] = aggActualAdmin1['admin_1']
        aggActualAdmin1['fnid'] = aggActualAdmin1['fnid'].map(fnidMap)
    if admin_level == 3:
        aggActualAdmin1['admin_3']='None'
        aggActualAdmin1['name'] = aggActualAdmin1['admin_2']
        aggYield1['admin_3']='None'
        aggYield1['name'] = aggYield1['admin_2']
        aggYield1['indicator'] ='Yield'
        aggYield1['fnid'] = aggYield1['admin_2']
        aggYield1['fnid'] = aggYield1['fnid'].map(fnidMap)
        aggActualAdmin1['fnid'] = aggActualAdmin1['admin_2']
        aggActualAdmin1['fnid'] = aggActualAdmin1['fnid'].map(fnidMap)
    return aggYield1,aggActualAdmin1

def merge_admin1(df,admin_level,country,country_iso,fnidMap):

    newDF = pd.DataFrame(
        columns=['fnid','country','country_code', 'admin_1','admin_2', 'name', 'product','season_name','planting_year','planting_month', 'harvest_year', 'harvest_month', 'crop_production_system', 'indicator',
                 'value'])
    aggYield1,aggActualAdmin1=agg_admin1(df,admin_level,country,country_iso,fnidMap)
    df = pd.read_hdf('./data/crop/adm_crop_production_'+country_iso+'_admin1.hdf')
    df = df[df['gscd_code'] == 'calibrated']
    if country == 'Tanzania':
        df['country'].replace('Tanzania, United Republic of', 'Tanzania', inplace=True)
    df=pd.concat([df,aggActualAdmin1])
    for crop in df['product'].unique():
        couDF = df[(df['country']==country)&(df['product']==crop)]
        if len(couDF) == 0:
            continue
        cropSys = couDF.crop_production_system.unique()
        for crop_sys in couDF.crop_production_system.unique():
            for season_name in couDF[couDF['crop_production_system']==crop_sys].season_name.unique():
                csDF = couDF[(couDF.season_name == season_name)&(couDF.crop_production_system==crop_sys)]
                admin1=csDF.admin_1.unique()
                admin2 = csDF.admin_2.unique()
                if admin_level == 2:
                    for admin in admin1:
                        admin1DF= csDF[ csDF['admin_1']==admin]
                        admin1DF.sort_values(by='harvest_year',inplace=True)
                        newDF=pd.concat([newDF,admin1DF])
                if admin_level==3:
                    for admin in admin2:
                        admin2DF= csDF[ csDF['admin_2']==admin]
                        admin2DF.sort_values(by='harvest_year',inplace=True)
                        newDF=pd.concat([newDF,admin2DF])
    newDF.reset_index(inplace=True)
    newDF.drop(columns='index', axis=1, inplace=True)
    newDF['fnid'] =  newDF['admin_1']
    newDF['fnid'] =  newDF['fnid'].map(fnidMap)
    return newDF

def cleanPA(df_o,admin_level,country,country_iso,Ocrop_list, duplicate_positions,non_repeated_positions):
    newCroDF = pd.DataFrame()
    for positions in non_repeated_positions:
        cropDF = df_o[(df_o['product'] == Ocrop_list[positions])]
        newCroDF=pd.concat([newCroDF,cropDF])
    for positions in duplicate_positions:
        cropDF = df_o[(df_o['product'] == Ocrop_list[positions])]
        cropSys = cropDF.crop_production_system.unique()
        for crop_sys in cropSys:
            for season_name in cropDF[cropDF['crop_production_system'] == crop_sys].season_name.unique():
                csDF = cropDF[(cropDF.season_name == season_name) & (cropDF.crop_production_system == crop_sys)]
                csDF = csDF.drop_duplicates()
                # names = csDF.name.unique()
                admins=csDF[admin_level].unique()
                for admin in admins:
                    for season_year in csDF[csDF[admin_level] == admin].season_year.unique():
                        iDF = csDF[(csDF[admin_level] == admin) & (csDF['season_year'] == season_year)]
                        if (len(iDF) == 3):
                            newCroDF = pd.concat([iDF, newCroDF])
                            continue
                        if (len(iDF) == 1):
                            continue
                        # if (len(iDF) == 4):
                        #     subCrop1 = iDF[iDF['product'] == 'Millet (Bulrush)']
                        #     subCrop2 = iDF[iDF['product'] == 'Millet (Finger)']
                        #     if len(subCrop2) == 3:
                        #         newMilDF = pd.concat([subCrop2, newMilDF])
                        #     else:
                        #         newMilDF = pd.concat([subCrop1, newMilDF])
                        if (len(iDF) == 6) | (len(iDF) == 5):
                            print('There are two area values or two production values')
                        if len(iDF) == 2:
                            if len(iDF.indicator.unique()) == 1:
                                continue
                            else:
                                newCroDF = pd.concat([iDF, newCroDF])
    df = newCroDF
    return df

def combine(df,crop_list, duplicate_positions,non_repeated_positions):
    for positions in duplicate_positions:
        crop=crop_list[positions]
        for season_name in df[df['product'] == crop]['season_name'].unique():
            xcz = df[(df['product'] == crop) & (df['season_name'] == season_name)]['name'].unique()
            for admin in df[(df['product'] == crop) & (df['season_name'] == season_name)]['name'].unique():
                for year in df[(df['product'] == crop) & (df['season_name'] == season_name) & (df['name'] == admin)][
                    'harvest_year'].unique():
                    df.loc[(df['product'] == crop) & (df['name'] == admin) & (df['season_name'] == season_name) & (
                                df['harvest_year'] == year) & (df['indicator'] == 'yield'), 'value'].values[0] = df.loc[(df[
                                                                                                                             'product'] == crop) & (
                                                                                                                                  df[
                                                                                                                                        'name'] == admin) & (
                                                                                                                                    df[
                                                                                                                                        'season_name'] == season_name) & (
                                                                                                                                    df[
                                                                                                                                        'harvest_year'] == year) & (
                                                                                                                                    df[
                                                                                                                                        'indicator'] == 'production'), 'value'].values[
                                                                                                                 0] / \
                                                                                                             df.loc[(df[
                                                                                                                         'product'] == crop) & (
                                                                                                                                df[
                                                                                                                                    'name'] == admin) & (
                                                                                                                                df[
                                                                                                                                    'season_name'] == season_name) & (
                                                                                                                                df[
                                                                                                                                    'harvest_year'] == year) & (
                                                                                                                                df[
                                                                                                                                    'indicator'] == 'area'), 'value'].values[
                                                                                                                 0]
                    return df


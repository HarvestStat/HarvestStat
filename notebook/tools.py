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


def FDW_PD_Sweeper(df, area_priority='Area Harvested'):
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
        if row['fnid_short'][-2] == 'R': continue
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
        fdx = gdf.FNID.apply(lambda x: x[7] == '1'); gdf.loc[fdx,'name'] = gdf.loc[fdx, 'ADMIN1']
        fdx = gdf.FNID.apply(lambda x: x[7] == '2'); gdf.loc[fdx,'name'] = gdf.loc[fdx, 'ADMIN2']
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
        assert np.isclose(area_new.sum(1), area.sum(1)).all() == True
        assert np.isclose(prod_new.sum(1), prod.sum(1)).all() == True
    return area_new, prod_new

def FDW_PD_CaliSeasonYear(stack, df, link_ratio, cs, cy):
    # Change planting and Harvest season and year
    if len(cs) > 0:
        for s in cs:
            for m in cs[s]:
                for k, v in cs[s][m].items():
                    stack.loc[(stack['season_name']==s), m] = stack.loc[(stack['season_name']==s), m].replace({k:v})
                    df.loc[(stack['season_name']==s), m] = df.loc[(stack['season_name']==s), m].replace({k:v})
    if len(cy) > 0:
        for s in cy:
            for y, t in cy[s].items():
                stack.loc[stack['season_name']==s, y] += t
                df.loc[stack['season_name']==s, y] += t
    # Calibrate 'link_ratio'
    if len(cs) > 0:
        for fnid, ratio in link_ratio.items():
            ratio = link_ratio[fnid]
            mdx = ratio.index
            mdx = mdx.to_frame().reset_index(drop=True)
            for s in cs:
                for m in cs[s]:
                    for k, v in cs[s][m].items():
                        mdx.loc[mdx['season_name']==s, m] = mdx.loc[mdx['season_name']==s, m].replace(k,v)
            ratio.index = pd.MultiIndex.from_frame(mdx)
            link_ratio[fnid] = ratio
    return stack, df, link_ratio


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
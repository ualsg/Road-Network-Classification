import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# Function for making directory
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else: pass


# Configure CRHD format (street widths and colors)
street_types = ['service', 'residential', 'tertiary_link', 'tertiary', 'secondary_link', 'primary_link',
                 'motorway_link', 'secondary', 'trunk', 'primary', 'motorway']
#major_roads = ['secondary', 'trunk', 'primary', 'motorway']
#minor_roads = ['secondary', 'service', 'residential', 'tertiary_link', 'tertiary', 'secondary_link', 'primary_link', 'motorway_link']

street_widths = {'service' : 1,
                 'residential' : 1,
                 'tertiary_link':1,
                 'tertiary': 2,
                 'secondary_link': 2,
                 'primary_link':2,
                 'motorway_link':2,
                 'secondary': 3,
                 'trunk': 4,
                 'primary' : 4,
                 'motorway' : 2.5}

# Colors for CRHDs with black background
street_colors_b = {'service' : 'blue',
                 'residential' : 'blue',
                 'tertiary_link': 'blue',
                 'tertiary':'cornflowerblue',
                 'secondary_link': 'cornflowerblue',
                 'primary_link':'cornflowerblue',
                 'motorway_link':'cornflowerblue',
                 'trunk_link':'cornflowerblue',
                 'secondary': 'lightblue',
                 'trunk': 'white',
                 'primary' : 'white',
                 'motorway' : 'lightgrey'}

# Colors for CRHDs with white background
street_colors_w = {'service' : 'skyblue',
                 'residential' : 'skyblue',
                 'tertiary_link': 'skyblue',
                 'tertiary':'cornflowerblue',
                 'secondary_link': 'cornflowerblue',
                 'primary_link':'cornflowerblue',
                 'trunk_link':'cornflowerblue',
                 'motorway_link':'darkred',
                 'secondary': 'darkblue',
                 'trunk': 'black',
                 'primary' : 'black',
                 'motorway' : 'darkred'}


def PlotCRHD(center_point, dist, name=None, save_path=None, dpi=300, format='png'):
    '''
    Plot the CRHD for a given central coordinate and radius.

    :param center_point: Tuple of central coordinates -> (lng, lat).
    :param dist: Radius in meter -> int.
    :param name: Name of the plotted urban zone -> str.
    :param save_path: The save path of the CRHD image. If None, the CRHD would be shown inline.
    :param dpi: Image dpi.
    :param format: Image format.
    :return: If save_path is None, return the CRHD image. If save_path is not None, return None.
    '''

    G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
    for stype in street_types:
        if (gdf.highway==stype).any():
            gdf[gdf.highway==stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])

    plt.axis('off')
    if save_path:
        filename = os.path.join(save_path, f'{str(dist)}_{name}.{format}')
        plt.savefig(filename, dpi=dpi, bbox_inches='tight',pad_inches=0, format=format)
        plt.close()
    else:
        #plt.show()
        return plt.gcf()

def PlotCRHD_grid(idx, center_point, dist, save_path, dpi=300, format='png'):

    filename = os.path.join(save_path, f'{idx}.{format}')
    G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
    for stype in street_types:
        if (gdf.highway==stype).any():
            gdf[gdf.highway==stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])

    plt.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight',pad_inches=0, format=format)
    plt.close()

def PlotList(grids, dist, save_path, dpi=300, format='png'):
    failures = list(grids.index)
    iter = 1
    while failures and iter<=3:
        f = []
        for _id in failures:
            try:
                PlotCRHD_grid(_id, grids.loc[_id].coord, dist, save_path, dpi=dpi, format=format)
            except:
                print(f'Round{str(iter)}: {_id} image generation failed')
                f.append(_id)
        failures = f
        iter += 1

def PlotCity(dist, grid_path, save_path, dpi=300, format='png'):
    '''
    # Plot CRHD images for a gridded city.

    :param dist: radius in meter -> int.
    :param grid_path: filepath of the grids Shapefile.
    :param save_path: filepath to save the CRHD images.
    :param dpi: image dpi.
    :param format: image format
    :return: None
    '''
    mkdir(save_path)
    grids = gpd.read_file(grid_path).set_index('id').to_crs(epsg=4326)
    grids['coord'] = [(centroid.y, centroid.x) for centroid in grids.geometry.centroid]
    PlotList(grids, dist, save_path, dpi, format)
    print('complete!')


# def PlotHierarchy(cityName, point, dist, img_folder, dpi=300, format='png'):
#     plt.style.use('dark_background')
#     majorPath = f'../Data/images/{img_folder}/major/{str(dist)}major_{cityName}.{format}'
#     minorPath = f'../Data/images/{img_folder}/minor/{str(dist)}minor_{cityName}.{format}'
#     G = ox.graph_from_point(center_point=point, network_type='all', dist=dist)
#     gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
#     gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)
#
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     # gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
#     for stype in major_roads:
#         if (gdf.highway == stype).any():
#             gdf[gdf.highway == stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_b[stype])
#
#     plt.axis('off')
#     plt.savefig(majorPath, dpi=dpi, bbox_inches='tight', pad_inches=0, format=format)
#     plt.close()
#
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     for stype in minor_roads:
#         if (gdf.highway == stype).any():
#             gdf[gdf.highway == stype].plot(ax=ax, linewidth=1.5, edgecolor='white')
#
#     plt.axis('off')
#     plt.savefig(minorPath, dpi=dpi, bbox_inches='tight', pad_inches=0, format=format)
#     plt.close()

if __name__ == '__main__':
    pass
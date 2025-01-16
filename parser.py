import pandas as pd

def parse_property(property_name, property_path):
    property_grid = []
    collect_data = False
    with open(property_path, 'r') as file:
        lines = file.read().split('\n')
        for line in lines:
            if line.strip().lower() == property_name.lower():
                collect_data = True
            elif collect_data:
                if line.strip() == '':
                    break  # Stop data collection on empty line
                line_values = [v for v in line.split() if v != '']
                for value in line_values:
                    if '*' in value:
                        count, value = value.split('*')
                        property_grid.extend([float(value)] * int(count))
                    else:
                        property_grid.append(float(value))
    return property_grid

def get_property_at_coordinates(grid, x, y, z, dim_x, dim_y):
    index = (x - 1) + (y - 1) * dim_x + (dim_x * dim_y * (z - 1))
    return grid[index]

def average_around_well_coordinates(poro_grid, wells_data, dim_x, dim_y, dim_z, radius):
    well_averages = {}
    for wname, group in wells_data.groupby('wname'):
        averages_by_level = [None] * dim_z
        last_x, last_y = group.iloc[-1]['iw'], group.iloc[-1]['jw']

        for z in range(1, dim_z + 1):
            sum_values = 0
            count = 0
            level_covered = False

            for _, row in group.iterrows():
                x, y = row['iw'], row['jw']
                if row['kw1'] <= z < row['kw2']:
                    level_covered = True
                    for dy in range(max(1, y - radius), min(dim_y + 1, y + radius + 1)):
                        for dx in range(max(1, x - radius), min(dim_x + 1, x + radius + 1)):
                            try:
                                value = get_property_at_coordinates(poro_grid, dx, dy, z, dim_x, dim_y)
                                sum_values += value
                                count += 1
                            except IndexError:
                                continue

            if not level_covered:
                for dy in range(max(1, last_y - radius), min(dim_y + 1, last_y + radius + 1)):
                    for dx in range(max(1, last_x - radius), min(dim_x + 1, last_x + radius + 1)):
                        try:
                            value = get_property_at_coordinates(poro_grid, dx, dy, z, dim_x, dim_y)
                            sum_values += value
                            count += 1
                        except IndexError:
                            continue

            averages_by_level[z - 1] = sum_values / count if count > 0 else None

        well_averages[wname] = averages_by_level

    return well_averages

if __name__ == '__main__':
    properties = {
        'PORO': 'Пористость.map',
        'INIT_PERMX': 'Проницаемость_по_X.map',
        'SOIL': 'map0.txt'
    }

    wells_data_path = 'wells_data.csv'
    dim_x, dim_y, dim_z = 139, 48, 9
    radii = [1, 2, 3]
    wells_data = pd.read_csv(wells_data_path)

    # Dictionary to store data in a structured way before DataFrame creation
    structured_data = {wname: {prop: [] for prop in properties} for wname in wells_data['wname'].unique()}

    for property_name, property_path in properties.items():
        property_grid = parse_property(property_name, property_path)

        for radius in radii:
            well_averages = average_around_well_coordinates(property_grid, wells_data, dim_x, dim_y, dim_z, radius)

            for wname, averages in well_averages.items():
                structured_data[wname][property_name].append(averages)

    # Creating DataFrame from structured_data
    final_data = []
    for wname, props in structured_data.items():
        well_info = wells_data[wells_data['wname'] == wname].iloc[0]
        row = {'Well Name': wname, 'Initial X': well_info['iw'], 'Initial Y': well_info['jw']}
        for prop, arrays in props.items():
            row[prop] = arrays  # Storing list of lists (three lists of nine values each)
        final_data.append(row)

    final_df = pd.DataFrame(final_data)

    output_file_path = 'all_well_averages.csv'
    final_df.to_csv(output_file_path, index=False)
    print(f"All well average data for all properties and radii saved to {output_file_path}")

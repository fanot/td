
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
    for index, row in wells_data.iterrows():
        wname = row['wname']
        x, y = row['iw'], row['jw']
        averages_by_level = []
        for z in range(1, dim_z + 1):  # Fixed range from 1 to dim_z
            sum_values = 0
            count = 0
            for dy in range(max(1, y - radius), min(dim_y + 1, y + radius + 1)):
                for dx in range(max(1, x - radius), min(dim_x + 1, x + radius + 1)):
                    try:
                        value = get_property_at_coordinates(poro_grid, dx, dy, z, dim_x, dim_y)
                        sum_values += value
                        count += 1
                    except IndexError:
                        continue
            if count > 0:
                averages_by_level.append(sum_values / count)
            else:
                averages_by_level.append(None)  
        well_averages[wname] = averages_by_level
    return well_averages

if __name__ == '__main__':
    property_name = 'PORO' # 
    property_path = 'Пористость.map'
    wells_data_path = 'wells_data.csv'
    dim_x, dim_y, dim_z = 139, 48, 9 

    wells_data = pd.read_csv(wells_data_path)
    poro_grid = parse_property(property_name, property_path)
    
    radius = 2  
    well_averages = average_around_well_coordinates(poro_grid, wells_data, dim_x, dim_y, dim_z, radius)
    print(f"Well average porosities for radius {radius}:", well_averages)

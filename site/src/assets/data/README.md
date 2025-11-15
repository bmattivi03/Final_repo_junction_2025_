# CSV Data Files

Place your CSV files here for marker and data loading.

## File Structure

```
src/assets/data/
  ├── markers.csv          # Marker data (locations, consumption, etc.)
  └── consumption.csv     # Consumption data
```

## CSV Format for Markers

Your `markers.csv` should have columns like:

```csv
id,name,subregion,city,groupId,lng,lat,predictedConsumption,trend
1,Joensuu,Pohjois-Karjala,Joensuu,28,29.76,62.60,645,up
2,Etelä-Savo,Etelä-Savo,Etelä-Savo,29,27.50,61.50,892,stable
...
```

**Column names can vary** - the loader will try to match common variations:
- `lng` / `longitude` / `lon`
- `lat` / `latitude`
- `predictedConsumption` / `consumption` / `predicted_consumption`
- `groupId` / `group_id`

## Usage

See `MapSection.tsx` for examples of loading CSV data.


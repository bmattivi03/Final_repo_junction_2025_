import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Button } from './ui/button';
import { SelectionState } from '../App';

interface FilterPanelProps {
  selection: SelectionState;
  onUpdateSelection: (updates: Partial<SelectionState>) => void;
  onVisualize: () => void;
}

const regions = [
  'Eastern Finland',
  'Lapland',
  'Southern Finland',
  'Western Finland',
];

const subregionsByRegion: Record<string, string[]> = {
  'Eastern Finland': ['Etelä-Savo', 'Pohjois-Karjala'],
  'Lapland': ['Lappi'],
  'Southern Finland': ['Uusimaa', 'Varsinais-Suomi'],
  'Western Finland': ['Pohjanmaa', 'Pirkanmaa'],
};

const citiesBySubregion: Record<string, string[]> = {
  'Etelä-Savo': ['Etelä-Savo', 'Mikkeli'],
  'Pohjois-Karjala': ['Joensuu', 'Pohjois-Karjala_Others'],
  'Lappi': ['Rovaniemi', 'Lappi_Others'],
  'Uusimaa': ['Helsinki', 'Espoo'],
  'Varsinais-Suomi': ['Turku', 'Varsinais-Suomi_Others'],
  'Pohjanmaa': ['Vaasa', 'Pohjanmaa_Others'],
  'Pirkanmaa': ['Tampere', 'Pirkanmaa_Others'],
};

// Map city names to group IDs
const cityGroupIds: Record<string, number> = {
  'Joensuu': 28,
  'Etelä-Savo': 29,
  'Rovaniemi': 30,
  'Lappi_Others': 31,
  'Helsinki': 32,
  'Turku': 33,
  'Tampere': 34,
  'Vaasa': 35,
  'Pohjois-Karjala_Others': 36,
  'Mikkeli': 37,
  'Espoo': 38,
  'Varsinais-Suomi_Others': 39,
  'Pohjanmaa_Others': 40,
  'Pirkanmaa_Others': 41,
};

// Map subregion to region for automatic region assignment
const subregionToRegion: Record<string, string> = {
  'Etelä-Savo': 'Eastern Finland',
  'Pohjois-Karjala': 'Eastern Finland',
  'Lappi': 'Lapland',
  'Uusimaa': 'Southern Finland',
  'Varsinais-Suomi': 'Southern Finland',
  'Pohjanmaa': 'Western Finland',
  'Pirkanmaa': 'Western Finland',
};

export function FilterPanel({ selection, onUpdateSelection, onVisualize }: FilterPanelProps) {
  const availableSubregions = selection.region ? subregionsByRegion[selection.region] || [] : [];
  const availableCities = selection.subregion ? citiesBySubregion[selection.subregion] || [] : [];

  const handleCityChange = (city: string) => {
    const groupId = cityGroupIds[city] || null;
    // Ensure region is set based on current subregion
    const region = selection.subregion ? subregionToRegion[selection.subregion] : selection.region;
    onUpdateSelection({ region, city, groupId });
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-slate-700 mb-2">Select Region</label>
          <Select
            value={selection.region}
            onValueChange={(value) => onUpdateSelection({ region: value, subregion: '', city: '' })}
          >
            <SelectTrigger className="bg-slate-50 border-slate-300">
              <SelectValue placeholder="Choose region..." />
            </SelectTrigger>
            <SelectContent>
              {regions.map((region) => (
                <SelectItem key={region} value={region}>
                  {region}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <label className="block text-slate-700 mb-2">Select Subregion</label>
          <Select
            value={selection.subregion}
            onValueChange={(value) => onUpdateSelection({ subregion: value, city: '' })}
            disabled={!selection.region}
          >
            <SelectTrigger className="bg-slate-50 border-slate-300">
              <SelectValue placeholder="Choose subregion..." />
            </SelectTrigger>
            <SelectContent>
              {availableSubregions.map((subregion) => (
                <SelectItem key={subregion} value={subregion}>
                  {subregion}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <label className="block text-slate-700 mb-2">Select City / Group</label>
          <Select
            value={selection.city}
            onValueChange={handleCityChange}
            disabled={!selection.subregion}
          >
            <SelectTrigger className="bg-slate-50 border-slate-300">
              <SelectValue placeholder="Choose city..." />
            </SelectTrigger>
            <SelectContent>
              {availableCities.map((city) => (
                <SelectItem key={city} value={city}>
                  {city}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Button
        onClick={onVisualize}
        disabled={!selection.city}
        className="w-full md:w-auto bg-[#2A66FF] hover:bg-[#1f4fd4]"
      >
        Visualize
      </Button>
    </div>
  );
}
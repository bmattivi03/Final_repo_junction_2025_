/**
 * Utility functions for loading and parsing CSV files
 */

export interface CSVRow {
  [key: string]: string | number;
}

export interface GroupTableRow {
  group_id: number;
  region: string;
  subregion: string;
  city: string;
  customerType: string;
  priceType: string;
  consumptionLevel: string;
}

/**
 * Load and parse a CSV file
 * @param url - Path to CSV file (relative to public or assets folder)
 * @param delimiter - CSV delimiter (default: ',', can be ';' for semicolon-delimited)
 * @returns Promise with parsed CSV data as array of objects
 */
export async function loadCSV(url: string, delimiter: string = ','): Promise<CSVRow[]> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load CSV: ${response.statusText}`);
    }
    const text = await response.text();
    return parseCSV(text, delimiter);
  } catch (error) {
    console.error(`Error loading CSV from ${url}:`, error);
    throw error;
  }
}

/**
 * Parse CSV text into array of objects
 * @param csvText - Raw CSV text content
 * @param delimiter - CSV delimiter (default: ',')
 * @returns Array of objects with keys from header row
 */
export function parseCSV(csvText: string, delimiter: string = ','): CSVRow[] {
  const lines = csvText.trim().split('\n');
  if (lines.length === 0) return [];

  // Parse header
  const headers = lines[0].split(delimiter).map(h => h.trim().replace(/^"|"$/g, ''));
  
  // Parse data rows
  const rows: CSVRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(delimiter).map(v => v.trim().replace(/^"|"$/g, ''));
    const row: CSVRow = {};
    headers.forEach((header, index) => {
      const value = values[index] || '';
      // Try to parse as number if possible
      const numValue = Number(value);
      row[header] = isNaN(numValue) || value === '' ? value : numValue;
    });
    rows.push(row);
  }
  
  return rows;
}

/**
 * Parse group table CSV (semicolon-delimited with pipe-separated group_label)
 * @param csvText - Raw CSV text content
 * @returns Array of parsed group table rows
 */
export function parseGroupTable(csvText: string): GroupTableRow[] {
  const lines = csvText.trim().split('\n');
  if (lines.length <= 1) return [];

  const rows: GroupTableRow[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const parts = line.split(';');
    if (parts.length < 2) continue;
    
    const groupId = Number(parts[0].trim());
    if (isNaN(groupId)) continue;
    
    const groupLabel = parts[1].trim();
    // Parse pipe-separated values: "Region | Subregion | City | CustomerType | PriceType | ConsumptionLevel"
    const labelParts = groupLabel.split('|').map(p => p.trim());
    
    rows.push({
      group_id: groupId,
      region: labelParts[0] || '',
      subregion: labelParts[1] || '',
      city: labelParts[2] || '',
      customerType: labelParts[3] || '',
      priceType: labelParts[4] || '',
      consumptionLevel: labelParts[5] || '',
    });
  }
  
  return rows;
}

/**
 * Load and parse group table CSV
 * @param url - Path to group table CSV file
 * @returns Promise with parsed group table data
 */
export async function loadGroupTable(url: string): Promise<GroupTableRow[]> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load group table: ${response.statusText}`);
    }
    const text = await response.text();
    return parseGroupTable(text);
  } catch (error) {
    console.error(`Error loading group table from ${url}:`, error);
    throw error;
  }
}

/**
 * Convert CSV rows to RegionMarker format
 * Adjust column names based on your CSV structure
 */
export function csvToMarkers(csvRows: CSVRow[]): Array<{
  id: string;
  name: string;
  subregion: string;
  city: string;
  groupId: number;
  lng: number;
  lat: number;
  predictedConsumption: number;
  trend: "up" | "down" | "stable";
}> {
  return csvRows.map((row, index) => ({
    id: String(row.id || index + 1),
    name: String(row.name || row.city || ''),
    subregion: String(row.subregion || ''),
    city: String(row.city || row.name || ''),
    groupId: Number(row.groupId || row.group_id || 0),
    lng: Number(row.lng || row.longitude || row.lon || 0),
    lat: Number(row.lat || row.latitude || 0),
    predictedConsumption: Number(row.predictedConsumption || row.consumption || row.predicted_consumption || 0),
    trend: (row.trend as "up" | "down" | "stable") || "stable",
  }));
}

/**
 * Parse forecast CSV (semicolon-delimited, comma as decimal separator)
 * @param csvText - Raw CSV text content
 * @param groupId - Group ID to extract data for
 * @param isMonthly - Whether this is monthly data (12m) or hourly (48h)
 * @returns Array of forecast data points
 */
export function parseForecastCSV(
  csvText: string, 
  groupId: number,
  isMonthly: boolean = false
): Array<{
  time: string;
  fullTime: string;
  consumption: number;
  day?: number;
  hour?: number;
  month?: string;
}> {
  const lines = csvText.trim().split('\n');
  if (lines.length <= 1) return [];

  const headers = lines[0].split(';').map(h => h.trim());
  const groupIdIndex = headers.findIndex(h => h === String(groupId));
  
  if (groupIdIndex === -1) {
    console.warn(`Group ID ${groupId} not found in forecast CSV`);
    return [];
  }

  const data: Array<{
    time: string;
    fullTime: string;
    consumption: number;
    day?: number;
    hour?: number;
    month?: string;
  }> = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const values = line.split(';');
    const timestamp = values[0]?.trim();
    const consumptionStr = values[groupIdIndex]?.trim() || '0';
    
    // Convert comma decimal separator to dot
    const consumption = parseFloat(consumptionStr.replace(',', '.'));

    if (!timestamp || isNaN(consumption)) continue;

    const date = new Date(timestamp);
    if (isNaN(date.getTime())) continue;

    if (isMonthly) {
      // Monthly data - format as month/year
      const monthStr = date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      data.push({
        month: monthStr,
        time: monthStr,
        fullTime: monthStr,
        consumption: Math.round(consumption * 100) / 100,
      });
    } else {
      // Hourly data - format with hour
      const day = Math.floor((i - 1) / 24) + 1;
      const hour = (i - 1) % 24;
      data.push({
        time: `${hour.toString().padStart(2, '0')}:00`,
        fullTime: date.toLocaleString('en-US', { 
          month: 'short', 
          day: 'numeric', 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        consumption: Math.round(consumption * 100) / 100,
        day,
        hour,
      });
    }
  }

  return data;
}

/**
 * Load forecast CSV data for a specific group
 * @param url - Path to forecast CSV file
 * @param groupId - Group ID to extract data for
 * @param isMonthly - Whether this is monthly data (12m) or hourly (48h)
 * @returns Promise with parsed forecast data
 */
export async function loadForecastData(
  url: string, 
  groupId: number,
  isMonthly: boolean = false
): Promise<Array<{
  time: string;
  fullTime: string;
  consumption: number;
  day?: number;
  hour?: number;
  month?: string;
}>> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load forecast CSV: ${response.statusText}`);
    }
    const text = await response.text();
    return parseForecastCSV(text, groupId, isMonthly);
  } catch (error) {
    console.error(`Error loading forecast CSV from ${url}:`, error);
    throw error;
  }
}


import { SelectionState } from '../App';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, Calendar, Zap, Clock } from 'lucide-react';
import { useState, useEffect } from 'react';
import { loadForecastData } from '../utils/csvLoader';

interface ChartSectionProps {
  selection: SelectionState;
}

// Custom tooltip for hourly data
const HourlyTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-4 rounded-lg shadow-xl border border-slate-200">
        <p className="text-slate-800 mb-2" style={{ fontWeight: 600 }}>{data.fullTime}</p>
        <p className="text-[#2A66FF] mb-1">
          <span style={{ fontWeight: 600 }}>Predicted:</span> {data.consumption.toLocaleString()} kWh
        </p>
        <p className="text-xs text-slate-500 mt-2">
          Day {data.day} • Hour {data.hour}
        </p>
      </div>
    );
  }
  return null;
};

// Custom tooltip for monthly data
const MonthlyTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-4 rounded-lg shadow-xl border border-slate-200">
        <p className="text-slate-800 mb-2" style={{ fontWeight: 600 }}>{data.month}</p>
        <p className="text-[#6A38FF] mb-1">
          <span style={{ fontWeight: 600 }}>Predicted:</span> {data.consumption.toLocaleString()} kWh
        </p>
      </div>
    );
  }
  return null;
};

export function ChartSection({ selection }: ChartSectionProps) {
  const [activeView, setActiveView] = useState<'hourly' | 'monthly'>('hourly');
  const [hourlyData, setHourlyData] = useState<Array<{
    time: string;
    fullTime: string;
    consumption: number;
    day?: number;
    hour?: number;
  }>>([]);
  const [monthlyData, setMonthlyData] = useState<Array<{
    month: string;
    consumption: number;
  }>>([]);
  const [loading, setLoading] = useState(false);

  // Load forecast data from CSV files when groupId changes
  useEffect(() => {
    if (!selection.groupId) {
      // Reset to empty if no group selected
      setHourlyData([]);
      setMonthlyData([]);
      return;
    }

    setLoading(true);

    // Load both CSV files - specify which is monthly
    Promise.all([
      loadForecastData(
        new URL('../assets/data/forecast_48h.csv', import.meta.url).href,
        selection.groupId,
        false // hourly
      ),
      loadForecastData(
        new URL('../assets/data/forecast_12m.csv', import.meta.url).href,
        selection.groupId,
        true // monthly
      ),
    ])
      .then(([hourly, monthly]) => {
        setHourlyData(hourly as Array<{
          time: string;
          fullTime: string;
          consumption: number;
          day?: number;
          hour?: number;
        }>);
        setMonthlyData(monthly.map(m => ({
          month: m.month || m.time,
          consumption: m.consumption,
        })));
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error loading forecast data:', error);
        setLoading(false);
        // Keep empty arrays on error
      });
  }, [selection.groupId]);

  return (
    <div className="space-y-6">
      {/* Header with Toggle */}
      <div className="border-b border-slate-200 pb-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h3 className="text-slate-900" style={{ fontSize: '1.25rem', fontWeight: 600 }}>
              Energy Consumption Forecast
            </h3>
            <p className="text-slate-600 mt-2" style={{ fontSize: '0.9375rem' }}>
              Predicted consumption for <strong>{selection.city || 'selected location'}</strong>
            </p>
            {/* Active Filters Display - Prominent */}
            {selection.filters && (
              selection.filters.customerType || selection.filters.priceType || selection.filters.consumptionLevel
            ) && (
              <div className="mt-3 flex flex-wrap gap-2">
                {selection.filters.customerType && (
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Customer: {selection.filters.customerType}
                  </span>
                )}
                {selection.filters.priceType && (
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    Price: {selection.filters.priceType}
                  </span>
                )}
                {selection.filters.consumptionLevel && (
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                    Consumption: {selection.filters.consumptionLevel}
                  </span>
                )}
              </div>
            )}
          </div>
          
          {/* View Toggle */}
          <div className="flex gap-2 bg-slate-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveView('hourly')}
              className={`px-4 py-2 rounded-md transition-all ${
                activeView === 'hourly'
                  ? 'bg-white text-[#2A66FF] shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              style={{ fontSize: '0.875rem', fontWeight: 600 }}
            >
              <Clock className="w-4 h-4 inline-block mr-2" />
              2-Day Hourly
            </button>
            <button
              onClick={() => setActiveView('monthly')}
              className={`px-4 py-2 rounded-md transition-all ${
                activeView === 'monthly'
                  ? 'bg-white text-[#6A38FF] shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              style={{ fontSize: '0.875rem', fontWeight: 600 }}
            >
              <Calendar className="w-4 h-4 inline-block mr-2" />
              12-Month
            </button>
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200">
        {loading ? (
          <div className="flex items-center justify-center h-[400px]">
            <p className="text-slate-600">Loading forecast data...</p>
          </div>
        ) : (activeView === 'hourly' ? hourlyData.length === 0 : monthlyData.length === 0) ? (
          <div className="flex items-center justify-center h-[400px]">
            <p className="text-slate-600">
              {selection.groupId 
                ? 'No forecast data available for this group'
                : 'Select a location on the map to view forecast data'}
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart
              data={activeView === 'hourly' ? hourlyData : monthlyData}
            margin={{ top: 10, right: 30, left: 20, bottom: 5 }}
          >
            <defs>
              <linearGradient id="colorHourly" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#2A66FF" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#2A66FF" stopOpacity={0.05}/>
              </linearGradient>
              <linearGradient id="colorMonthly" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6A38FF" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#6A38FF" stopOpacity={0.05}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis 
              dataKey={activeView === 'hourly' ? 'time' : 'month'}
              stroke="#64748b"
              style={{ fontSize: '11px' }}
              angle={activeView === 'hourly' ? -45 : 0}
              textAnchor={activeView === 'hourly' ? 'end' : 'middle'}
              height={activeView === 'hourly' ? 80 : 30}
              interval={activeView === 'hourly' ? 5 : 0}
            />
            <YAxis 
              stroke="#64748b"
              style={{ fontSize: '12px' }}
              label={{ 
                value: activeView === 'hourly' ? 'Hourly Consumption (kWh)' : 'Monthly Consumption (kWh)', 
                angle: -90, 
                position: 'insideLeft',
                style: { fill: '#64748b', fontSize: '14px', fontWeight: 500 }
              }}
            />
            <Tooltip content={activeView === 'hourly' ? <HourlyTooltip /> : <MonthlyTooltip />} />
            <Legend 
              wrapperStyle={{ paddingTop: '20px' }}
            />
            
            {/* Area under the prediction line */}
            <Area
              type="monotone"
              dataKey="consumption"
              stroke={activeView === 'hourly' ? '#2A66FF' : '#6A38FF'}
              strokeWidth={3}
              fill={activeView === 'hourly' ? 'url(#colorHourly)' : 'url(#colorMonthly)'}
              name={activeView === 'hourly' ? 'Hourly Prediction' : 'Monthly Prediction'}
            />
          </AreaChart>
        </ResponsiveContainer>
        )}
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {activeView === 'hourly' ? (
          <>
            <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100/50 rounded-lg border border-blue-200">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-blue-500 rounded-lg">
                  <Clock className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-slate-900 mb-1" style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
                    Hourly Granularity
                  </p>
                  <p className="text-slate-600" style={{ fontSize: '0.875rem', lineHeight: 1.6 }}>
                    48-hour forecast with hourly resolution for immediate operational planning and intraday load balancing.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-gradient-to-br from-cyan-50 to-cyan-100/50 rounded-lg border border-cyan-200">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-cyan-500 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-slate-900 mb-1" style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
                    Peak Detection
                  </p>
                  <p className="text-slate-600" style={{ fontSize: '0.875rem', lineHeight: 1.6 }}>
                    Identifies daily consumption patterns and peak demand hours for resource optimization.
                  </p>
                </div>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100/50 rounded-lg border border-purple-200">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-purple-500 rounded-lg">
                  <Calendar className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-slate-900 mb-1" style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
                    Annual Forecast
                  </p>
                  <p className="text-slate-600" style={{ fontSize: '0.875rem', lineHeight: 1.6 }}>
                    12-month strategic forecast for long-term capacity planning and seasonal trend analysis.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-gradient-to-br from-violet-50 to-violet-100/50 rounded-lg border border-violet-200">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-violet-500 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-slate-900 mb-1" style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
                    Seasonal Patterns
                  </p>
                  <p className="text-slate-600" style={{ fontSize: '0.875rem', lineHeight: 1.6 }}>
                    Captures seasonal consumption variations for budget forecasting and infrastructure planning.
                  </p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Model Info */}
      <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-slate-700 rounded-lg">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <p className="text-slate-900 mb-2" style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
              Prediction Model Details
            </p>
            <p className="text-slate-600 mb-3" style={{ fontSize: '0.875rem', lineHeight: 1.7 }}>
              Time-series forecasting model trained on historical consumption patterns, weather data, 
              pricing structures, and production statistics. Features include lag variables, rolling 
              averages, and temporal embeddings for enhanced accuracy.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-3 border-t border-slate-200">
              <div>
                <p className="text-slate-500 text-xs uppercase tracking-wide mb-1">Group ID</p>
                <p className="text-slate-900" style={{ fontSize: '1rem', fontWeight: 600 }}>
                  {selection.groupId || '--'}
                </p>
              </div>
              <div>
                <p className="text-slate-500 text-xs uppercase tracking-wide mb-1">Location</p>
                <p className="text-slate-900" style={{ fontSize: '1rem', fontWeight: 600 }}>
                  {selection.city}
                </p>
              </div>
              <div>
                <p className="text-slate-500 text-xs uppercase tracking-wide mb-1">Model Accuracy</p>
                <p className="text-slate-900" style={{ fontSize: '1rem', fontWeight: 600 }}>
                  94.2%
                </p>
              </div>
              <div>
                <p className="text-slate-500 text-xs uppercase tracking-wide mb-1">Forecast Period</p>
                <p className="text-slate-900" style={{ fontSize: '1rem', fontWeight: 600 }}>
                  {activeView === 'hourly' ? '48 Hours' : '12 Months'}
                </p>
              </div>
            </div>
            
            {/* Active Filters Display */}
            {selection.filters && (
              selection.filters.customerType || selection.filters.priceType || selection.filters.consumptionLevel
            ) && (
              <div className="mt-4 pt-4 border-t border-slate-200">
                <p className="text-slate-500 text-xs uppercase tracking-wide mb-3">Active Filters</p>
                <div className="flex flex-wrap gap-2">
                  {selection.filters.customerType && (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      Customer: {selection.filters.customerType}
                    </span>
                  )}
                  {selection.filters.priceType && (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Price: {selection.filters.priceType}
                    </span>
                  )}
                  {selection.filters.consumptionLevel && (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                      Consumption: {selection.filters.consumptionLevel}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
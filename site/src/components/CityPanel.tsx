import { SelectionState } from '../App';
import { Building2, Users, CreditCard, TrendingUp, Hash } from 'lucide-react';

interface CityPanelProps {
  selection: SelectionState;
}

// Mock data for demonstration
const cityData: Record<string, {
  customerType: string;
  pricingModel: string;
  consumptionTier: string;
  groupId: number;
}> = {
  'Joensuu': { customerType: 'Residential', pricingModel: 'Fixed Rate', consumptionTier: 'Medium', groupId: 28 },
  'Etelä-Savo': { customerType: 'Industrial', pricingModel: 'Dynamic', consumptionTier: 'High', groupId: 29 },
  'Rovaniemi': { customerType: 'Commercial', pricingModel: 'Time-of-Use', consumptionTier: 'Medium', groupId: 30 },
  'Lappi_Others': { customerType: 'Residential', pricingModel: 'Fixed Rate', consumptionTier: 'Low', groupId: 31 },
  'Helsinki': { customerType: 'Mixed', pricingModel: 'Dynamic', consumptionTier: 'High', groupId: 32 },
  'Espoo': { customerType: 'Residential', pricingModel: 'Fixed Rate', consumptionTier: 'High', groupId: 33 },
  'Turku': { customerType: 'Commercial', pricingModel: 'Time-of-Use', consumptionTier: 'Medium', groupId: 34 },
  'Tampere': { customerType: 'Mixed', pricingModel: 'Dynamic', consumptionTier: 'High', groupId: 35 },
  'Vaasa': { customerType: 'Industrial', pricingModel: 'Time-of-Use', consumptionTier: 'High', groupId: 36 },
  'Mikkeli': { customerType: 'Residential', pricingModel: 'Fixed Rate', consumptionTier: 'Medium', groupId: 37 },
  'Pohjois-Karjala_Others': { customerType: 'Residential', pricingModel: 'Fixed Rate', consumptionTier: 'Low', groupId: 38 },
};

export function CityPanel({ selection }: CityPanelProps) {
  const data = selection.city ? cityData[selection.city] : null;
  const displayGroupId = selection.groupId || data?.groupId || null;
  
  // Use filters from selection if available, otherwise fall back to mock data
  const customerType = selection.filters?.customerType || data?.customerType || null;
  const priceType = selection.filters?.priceType || data?.pricingModel || null;
  const consumptionLevel = selection.filters?.consumptionLevel || data?.consumptionTier || null;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200 h-fit">
      <h3 className="text-slate-900 mb-6 border-b border-slate-200 pb-3" style={{ fontSize: '1.25rem', fontWeight: 600 }}>
        Location Details
      </h3>

      <div className="space-y-4">
        <div className="flex items-start gap-3">
          <div className="mt-1 p-2 bg-[#2A66FF] bg-opacity-10 rounded-lg">
            <Building2 className="w-5 h-5 text-[#2A66FF]" />
          </div>
          <div className="flex-1">
            <p className="text-slate-500">Selected Region</p>
            <p className="text-slate-800">{selection.region || 'Not selected'}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <div className="mt-1 p-2 bg-[#6A38FF] bg-opacity-10 rounded-lg">
            <Building2 className="w-5 h-5 text-[#6A38FF]" />
          </div>
          <div className="flex-1">
            <p className="text-slate-500">Selected City</p>
            <p className="text-slate-800">{selection.city || 'Not selected'}</p>
          </div>
        </div>

        {selection.subregion && (
          <div className="flex items-start gap-3">
            <div className="mt-1 p-2 bg-indigo-100 rounded-lg">
              <Building2 className="w-5 h-5 text-indigo-600" />
            </div>
            <div className="flex-1">
              <p className="text-slate-500">Subregion</p>
              <p className="text-slate-800">{selection.subregion}</p>
            </div>
          </div>
        )}

        {customerType && (
          <div className="flex items-start gap-3">
            <div className="mt-1 p-2 bg-blue-100 rounded-lg">
              <Users className="w-5 h-5 text-blue-600" />
            </div>
            <div className="flex-1">
              <p className="text-slate-500">Customer Type</p>
              <p className="text-slate-800">{customerType}</p>
            </div>
          </div>
        )}

        {priceType && (
          <div className="flex items-start gap-3">
            <div className="mt-1 p-2 bg-green-100 rounded-lg">
              <CreditCard className="w-5 h-5 text-green-600" />
            </div>
            <div className="flex-1">
              <p className="text-slate-500">Pricing Model</p>
              <p className="text-slate-800">{priceType}</p>
            </div>
          </div>
        )}

        {consumptionLevel && (
          <div className="flex items-start gap-3">
            <div className="mt-1 p-2 bg-purple-100 rounded-lg">
              <TrendingUp className="w-5 h-5 text-purple-600" />
            </div>
            <div className="flex-1">
              <p className="text-slate-500">Consumption Tier</p>
              <p className="text-slate-800">{consumptionLevel}</p>
            </div>
          </div>
        )}

        {displayGroupId && (
          <div className="flex items-start gap-3 pt-4 border-t border-slate-200">
            <div className="mt-1 p-2 bg-amber-100 rounded-lg">
              <Hash className="w-5 h-5 text-amber-600" />
            </div>
            <div className="flex-1">
              <p className="text-slate-500">Group ID</p>
              <p className="text-slate-800">{displayGroupId}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
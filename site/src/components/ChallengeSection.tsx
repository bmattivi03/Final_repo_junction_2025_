export function ChallengeSection() {
  return (
    <section className="py-24 px-6 bg-white">
      <div className="max-w-3xl mx-auto">
        {/* The Challenge */}
        <div className="mb-20">
          <h2 className="text-slate-900 mb-8 text-center tracking-tight" style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1.2 }}>
            The Challenge
          </h2>
          <div className="space-y-6 text-slate-600" style={{ fontSize: '1.0625rem', lineHeight: 1.8 }}>
            <p>
              Energy forecasting is a critical challenge for modern power grids. Accurate predictions of daily energy consumption 
              enable utilities to optimize generation, reduce waste, and ensure grid stability. The Junction 2025 Energy Forecasting 
              Challenge tasks teams with building predictive models that can forecast consumption patterns across diverse regions, 
              customer types, and market conditions.
            </p>
            <p>
              This challenge requires integrating multiple data sources—historical consumption, weather patterns, energy prices, 
              and production statistics—into a unified forecasting system. The complexity lies not just in model accuracy, but in 
              creating interpretable, actionable insights that can guide real-world energy management decisions across Finland's 
              varied geographic and demographic landscape.
            </p>
          </div>
        </div>

        {/* Insight */}
        <div className="mb-20">
          <h3 className="text-slate-900 mb-8 text-center tracking-tight" style={{ fontSize: '2rem', fontWeight: 600, lineHeight: 1.3 }}>
            Insight
          </h3>
          <p className="text-slate-600" style={{ fontSize: '1.0625rem', lineHeight: 1.8 }}>
            Accurate energy forecasting reduces operational costs, minimizes environmental impact through better resource allocation, 
            and enhances grid reliability. By understanding consumption patterns at granular levels—from individual cities to customer 
            segments—energy providers can proactively manage supply-demand dynamics, implement targeted efficiency programs, and 
            transition more effectively to renewable energy sources. This dashboard demonstrates how machine learning models can 
            transform raw data into strategic intelligence for Finland's energy future.
          </p>
        </div>

        {/* Resources */}
        <div>
          <h3 className="text-slate-900 mb-8 text-center tracking-tight" style={{ fontSize: '2rem', fontWeight: 600, lineHeight: 1.3 }}>
            Resources
          </h3>
          <div className="bg-gradient-to-br from-slate-50 to-blue-50/50 rounded-2xl p-8 border border-slate-200">
            <p className="text-slate-700 mb-6" style={{ fontSize: '1rem', fontWeight: 500 }}>
              The forecasting system leverages the following data sources:
            </p>
            <ul className="space-y-4 text-slate-600" style={{ fontSize: '1rem', lineHeight: 1.7 }}>
              <li className="flex gap-3">
                <span className="text-[#2A66FF] flex-shrink-0 self-start" style={{ fontSize: '1rem', fontWeight: 700 }}>•</span>
                <span><strong className="text-slate-800">Historical Consumption:</strong> Daily energy usage patterns across 50+ regions and customer segments</span>
              </li>
              <li className="flex gap-3">
                <span className="text-[#2A66FF] flex-shrink-0 self-start" style={{ fontSize: '1rem', fontWeight: 700 }}>•</span>
                <span><strong className="text-slate-800">Weather Data:</strong> Temperature, precipitation, and seasonal indicators from meteorological stations</span>
              </li>
              <li className="flex gap-3">
                <span className="text-[#2A66FF] flex-shrink-0 self-start" style={{ fontSize: '1rem', fontWeight: 700 }}>•</span>
                <span><strong className="text-slate-800">Energy Prices:</strong> Market pricing models including fixed-rate, time-of-use, and dynamic pricing structures</span>
              </li>
              <li className="flex gap-3">
                <span className="text-[#2A66FF] flex-shrink-0 self-start" style={{ fontSize: '1rem', fontWeight: 700 }}>•</span>
                <span><strong className="text-slate-800">Production Statistics:</strong> Generation capacity and renewable energy integration metrics</span>
              </li>
              <li className="flex gap-3">
                <span className="text-[#2A66FF] flex-shrink-0 self-start" style={{ fontSize: '1rem', fontWeight: 700 }}>•</span>
                <span><strong className="text-slate-800">Feature Engineering:</strong> Lag features, rolling averages, and temporal embeddings for enhanced predictions</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
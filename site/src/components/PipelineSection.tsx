import { InteractivePipelineViewer } from './InteractivePipelineViewer';

export function PipelineSection() {
  return (
    <section className="py-24 px-6 bg-slate-50">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-slate-900 mb-6 tracking-tight" style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1.2 }}>
            Our Data & Modeling Pipeline
          </h2>
          <p className="text-slate-600 max-w-3xl mx-auto" style={{ fontSize: '1.125rem', lineHeight: 1.7 }}>
            The forecasting system combines weather, prices, energy production, and feature engineering into a unified modeling pipeline.
          </p>
        </div>

        {/* Interactive Pipeline Viewer */}
        <div className="relative">
          <InteractivePipelineViewer />

          {/* Decorative Elements */}
          <div className="absolute -top-4 -left-4 w-24 h-24 bg-[#2A66FF] rounded-full opacity-5 blur-2xl pointer-events-none"></div>
          <div className="absolute -bottom-4 -right-4 w-32 h-32 bg-[#6A38FF] rounded-full opacity-5 blur-2xl pointer-events-none"></div>
        </div>

        {/* Pipeline Description */}
        <div className="mt-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div className="bg-white rounded-xl p-6 shadow-md border border-slate-200 transition-all duration-300 hover:shadow-xl hover:scale-105 hover:border-[#2A66FF] group">
              <h4 className="text-slate-900 mb-3 transition-colors duration-300 group-hover:text-[#2A66FF]" style={{ fontSize: '1.125rem', fontWeight: 600 }}>Input Layer</h4>
              <p className="text-slate-600 transition-colors duration-300 group-hover:text-slate-800" style={{ fontSize: '0.9375rem', lineHeight: 1.7 }}>
                Historical consumption, weather data, and pricing models are collected and validated from multiple sources.
              </p>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-md border border-slate-200 transition-all duration-300 hover:shadow-xl hover:scale-105 hover:border-[#5A4FFF] group">
              <h4 className="text-slate-900 mb-3 transition-colors duration-300 group-hover:text-[#5A4FFF]" style={{ fontSize: '1.125rem', fontWeight: 600 }}>Processing Layer</h4>
              <p className="text-slate-600 transition-colors duration-300 group-hover:text-slate-800" style={{ fontSize: '0.9375rem', lineHeight: 1.7 }}>
                Features are engineered with lag variables, rolling statistics, and temporal embeddings for enhanced accuracy.
              </p>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-md border border-slate-200 transition-all duration-300 hover:shadow-xl hover:scale-105 hover:border-[#6A38FF] group">
              <h4 className="text-slate-900 mb-3 transition-colors duration-300 group-hover:text-[#6A38FF]" style={{ fontSize: '1.125rem', fontWeight: 600 }}>Output Layer</h4>
              <p className="text-slate-600 transition-colors duration-300 group-hover:text-slate-800" style={{ fontSize: '0.9375rem', lineHeight: 1.7 }}>
                Trained models generate 30-day forecasts with confidence intervals and comprehensive accuracy metrics.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
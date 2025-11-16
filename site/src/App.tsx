import React, { useState } from 'react';
import { motion, useScroll, useTransform } from 'motion/react';
import { Header } from './components/Header';
import { ChallengeSection } from './components/ChallengeSection';
import { PipelineSection } from './components/PipelineSection';
import { MapSection } from './components/MapSection';
import { CityPanel } from './components/CityPanel';
import { ChartSection } from './components/ChartSection';
import { Footer } from './components/Footer';

export interface SelectionState {
  region: string;
  subregion: string;
  city: string;
  groupId: number | null;
  viewMode?: 'hourly' | 'monthly';
  filters?: {
    customerType: string;
    priceType: string;
    consumptionLevel: string;
  };
}

function App() {
  const [selection, setSelection] = useState<SelectionState>({
    region: '',
    subregion: '',
    city: '',
    groupId: null,
  });

  const [showVisualization, setShowVisualization] = useState(false);

  const { scrollY } = useScroll();
  
  // Transform scroll position for content fade in
  const contentOpacity = useTransform(scrollY, [200, 500], [0, 1]);
  const contentY = useTransform(scrollY, [200, 500], [50, 0]);

  const handleVisualize = () => {
    setShowVisualization(true);
    // Smooth scroll to map section
    setTimeout(() => {
      const mapSection = document.getElementById('map-section');
      if (mapSection) {
        mapSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  const handleMapMarkerClick = (
    region: string, 
    subregion: string, 
    city: string, 
    groupId: number,
    filters?: {
      customerType: string;
      priceType: string;
      consumptionLevel: string;
    },
    viewMode?: 'hourly' | 'monthly'
  ) => {
    setSelection({
      ...selection,
      region,
      subregion,
      city,
      groupId,
      filters,
      viewMode,
    });
    setShowVisualization(true);
    // Smooth scroll to graph section after state update
    setTimeout(() => {
      const graphSection = document.getElementById('graph-section');
      if (graphSection) {
        graphSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 200);
  };

  const updateSelection = (updates: Partial<SelectionState>) => {
    setSelection(prev => ({ ...prev, ...updates }));
    // If city is being updated and it's not empty, trigger visualization
    if (updates.city && updates.city !== '') {
      setShowVisualization(true);
      // Smooth scroll to map section
      setTimeout(() => {
        const mapSection = document.getElementById('map-section');
        if (mapSection) {
          mapSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* 1. HERO SECTION */}
      <Header />
      
      <motion.div 
        style={{ opacity: contentOpacity, y: contentY }}
      >
        {/* 2. CHALLENGE INTRO SECTION */}
        <ChallengeSection />

        {/* 3. INTERACTIVE MAP SECTION */}
        <section id="map-section" className="py-20 px-6 bg-slate-50">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-slate-900 mb-6 tracking-tight" style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1.2 }}>
                Geographical Overview
              </h2>
              <p className="text-slate-600 max-w-2xl mx-auto" style={{ fontSize: '1.125rem', lineHeight: 1.7 }}>
                Select a region on the map to explore predicted consumption patterns across Finland.
              </p>
            </div>
            
            <MapSection
              selectedRegion={selection.region}
              selectedCity={selection.city}
              onMarkerClick={handleMapMarkerClick}
            />
          </div>
        </section>

        {/* 4. INFORMATION PANEL + GRAPH SECTION */}
        {showVisualization && selection.city && selection.groupId && (
          <section id="graph-section" className="py-20 px-6 bg-white">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-slate-900 mb-6 tracking-tight" style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1.2 }}>
                  Daily Consumption Trends
                </h2>
                <p className="text-slate-600 max-w-2xl mx-auto" style={{ fontSize: '1.125rem', lineHeight: 1.7 }}>
                  View detailed consumption predictions and model performance metrics for the selected location.
                </p>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* LEFT: Info Panel */}
                <div className="lg:col-span-1">
                  <CityPanel selection={selection} />
                </div>

                {/* RIGHT: Charts */}
                <div className="lg:col-span-2">
                  <ChartSection selection={selection} />
                </div>
              </div>
            </div>
          </section>
        )}

        {/* 5. DATA & MODEL PIPELINE SECTION */}
        <PipelineSection />

        {/* 6. FOOTER SECTION */}
        <Footer />
      </motion.div>
    </div>
  );
}

export default App;
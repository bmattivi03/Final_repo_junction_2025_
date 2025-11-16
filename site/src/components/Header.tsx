import { motion, useScroll, useTransform } from 'motion/react';
import { ChevronDown } from 'lucide-react';
import { Button } from './ui/button';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface HeaderProps {
  onExploreClick?: () => void;
}

export function Header({ onExploreClick }: HeaderProps) {
  const { scrollY } = useScroll();
  
  // Transform scroll position to opacity (fades out as you scroll)
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);
  
  // Transform scroll position to scale (zooms in slightly as you scroll)
  const scale = useTransform(scrollY, [0, 400], [1, 1.1]);
  
  // Transform scroll position for text (moves up and fades)
  const heroY = useTransform(scrollY, [0, 400], [0, -100]);
  const textOpacity = useTransform(scrollY, [0, 300], [1, 0]);

  const handleExploreClick = () => {
    if (onExploreClick) {
      onExploreClick();
    } else {
      // Smooth scroll to content
      window.scrollTo({
        top: window.innerHeight,
        behavior: 'smooth'
      });
    }
  };

  const backgroundImage = 'https://i.imgur.com/rT2IGYs.jpeg';

  return (
    <header className="relative h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image with Overlay */}
      <motion.div
        className="absolute inset-0 z-0"
        style={{ opacity: heroOpacity }}
      >
        <ImageWithFallback
          src={backgroundImage}
          alt="Finland Energy Landscape"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900/90 via-slate-900/80 to-[#2A66FF]/40"></div>
      </motion.div>

      {/* Hero Content */}
      <motion.div
        className="relative z-10 text-center px-6 max-w-4xl"
        style={{ opacity: heroOpacity, y: heroY }}
      >
        {/* Glassmorphic Container */}
        <div className="backdrop-blur-md bg-white/10 rounded-3xl p-12 border border-white/20 shadow-2xl">
          {/* Main Title */}
          <h1 className="text-white mb-6 tracking-tight" style={{ fontSize: '3.5rem', fontWeight: 700, lineHeight: 1.1 }}>
          WattAhead
          </h1>
          <div className="h-1 w-24 bg-gradient-to-r from-[#2A66FF] to-[#6A38FF] mx-auto mb-6 rounded-full"></div>
          
          {/* Subtitle */}
          <p className="text-white/90 mb-4" style={{ fontSize: '1.5rem', fontWeight: 500, lineHeight: 1.4 }}>
            Finland
          </p>

          {/* Description */}
          <p className="text-white/80 max-w-2xl mx-auto mb-10" style={{ fontSize: '1.125rem', lineHeight: 1.7, fontWeight: 400 }}>
            Advanced ML solution for electricity consumption forecasting across 112 customer groups in Finland. 
            Predicts 48-hour hourly and 12-month monthly consumption using feature engineering, deep learning, and gradient boosting.
          </p>

          {/* CTA Button */}
          <button
            onClick={handleExploreClick}
            className="bg-gradient-to-r from-[#2A66FF] to-[#6A38FF] text-white px-8 py-4 rounded-xl hover:shadow-2xl hover:scale-105 transition-all duration-300"
            style={{ fontSize: '1.125rem', fontWeight: 600 }}
          >
            Start Exploring Data
          </button>
        </div>
      </motion.div>

      {/* Scroll Indicator */}
      <motion.div
        className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10"
        style={{ opacity: heroOpacity }}
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <div className="flex flex-col items-center gap-2 text-white/80">
          <span className="text-sm uppercase tracking-wider" style={{ fontWeight: 500 }}>Scroll to Explore</span>
          <ChevronDown className="w-6 h-6" />
        </div>
      </motion.div>
    </header>
  );
}
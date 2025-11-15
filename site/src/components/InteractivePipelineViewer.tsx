import { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Move } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';
import pipelineImage from '@/assets/images/pipeline.webp';

export function InteractivePipelineViewer() {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLDivElement>(null);

  const minScale = 0.5;
  const maxScale = 3;
  const zoomStep = 0.2;

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + zoomStep, maxScale));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - zoomStep, minScale));
  };

  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) { // Left click only
      e.preventDefault();
      setIsDragging(true);
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      e.preventDefault();
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      e.preventDefault();
      setIsDragging(true);
      setDragStart({
        x: e.touches[0].clientX - position.x,
        y: e.touches[0].clientY - position.y,
      });
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isDragging && e.touches.length === 1) {
      e.preventDefault();
      setPosition({
        x: e.touches[0].clientX - dragStart.x,
        y: e.touches[0].clientY - dragStart.y,
      });
    }
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    const handleMouseUpGlobal = () => setIsDragging(false);
    window.addEventListener('mouseup', handleMouseUpGlobal);
    return () => window.removeEventListener('mouseup', handleMouseUpGlobal);
  }, []);

  // Handle wheel zoom and prevent page scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      
      // Zoom based on wheel delta
      const delta = e.deltaY * -0.01;
      setScale(prev => {
        const newScale = prev + delta * 0.1;
        return Math.min(Math.max(newScale, minScale), maxScale);
      });
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, []);

  return (
    <div className="relative">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <button
          onClick={handleZoomIn}
          className="bg-white hover:bg-slate-50 text-slate-700 p-3 rounded-lg shadow-lg border border-slate-200 transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="w-5 h-5" />
        </button>
        <button
          onClick={handleZoomOut}
          className="bg-white hover:bg-slate-50 text-slate-700 p-3 rounded-lg shadow-lg border border-slate-200 transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="w-5 h-5" />
        </button>
        <button
          onClick={handleReset}
          className="bg-white hover:bg-slate-50 text-slate-700 p-3 rounded-lg shadow-lg border border-slate-200 transition-colors"
          title="Reset View"
        >
          <Maximize2 className="w-5 h-5" />
        </button>
      </div>

      {/* Zoom Level Indicator */}
      <div className="absolute top-4 left-4 z-10 bg-white px-4 py-2 rounded-lg shadow-lg border border-slate-200">
        <div className="flex items-center gap-2 text-sm text-slate-700">
          <Move className="w-4 h-4" />
          <span>{Math.round(scale * 100)}%</span>
        </div>
      </div>

      {/* Interactive Container */}
      <div
        ref={containerRef}
        className="bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden relative"
        style={{ height: '600px', cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {/* Image Container */}
        <div
          ref={imageRef}
          className="absolute inset-0 flex items-center justify-center p-8"
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            willChange: 'transform',
          }}
        >
          <ImageWithFallback
            src={pipelineImage}
            alt="Data & Modeling Pipeline"
            className="max-w-full h-auto rounded-lg pointer-events-none select-none"
            draggable={false}
          />
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 text-center text-sm text-slate-500">
        <p>💡 Click and drag to pan • Scroll to zoom • Use controls to navigate</p>
      </div>
    </div>
  );
}
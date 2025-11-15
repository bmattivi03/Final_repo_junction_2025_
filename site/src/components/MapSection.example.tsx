/**
 * EXAMPLE: How to load markers from CSV file
 * 
 * This is an example showing how to modify MapSection.tsx to load markers from CSV.
 * Copy the relevant parts into your MapSection.tsx file.
 */

import React, { useEffect, useRef, useState } from "react";
import maplibregl, { Map } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { loadCSV, csvToMarkers } from "@/utils/csvLoader";

// ... existing interfaces ...

export function MapSection({
  selectedRegion,
  selectedCity,
  onMarkerClick,
}: MapSectionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<Map | null>(null);
  const [mapLoaded, setLoaded] = useState(false);
  const [markers, setMarkers] = useState<RegionMarker[]>([]); // State for loaded markers

  // Load markers from CSV on component mount
  useEffect(() => {
    // Option 1: Load from public folder (files served as static assets)
    // const csvUrl = "/data/markers.csv";
    
    // Option 2: Load from assets folder (using Vite's asset handling)
    // You'll need to import the CSV or use a dynamic import
    const csvUrl = new URL("@/assets/data/markers.csv", import.meta.url).href;
    
    loadCSV(csvUrl)
      .then(csvRows => {
        const loadedMarkers = csvToMarkers(csvRows);
        setMarkers(loadedMarkers);
        console.log("Loaded markers from CSV:", loadedMarkers);
      })
      .catch(error => {
        console.error("Failed to load markers CSV:", error);
        // Fallback to hardcoded markers if CSV fails
        // setMarkers(defaultMarkers);
      });
  }, []);

  // Initialize map
  useEffect(() => {
    if (!ref.current) return;
    
    // ... existing map initialization code ...
    
    map.current.on("load", () => {
      setLoaded(true);
      
      // Use markers from state (loaded from CSV)
      markers.forEach((m) => {
        // ... existing marker creation code ...
      });
    });
  }, [markers]); // Re-run when markers change

  // ... rest of component ...
}


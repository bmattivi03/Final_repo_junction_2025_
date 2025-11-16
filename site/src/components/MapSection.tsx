import React, { useEffect, useRef, useState } from "react";
import maplibregl, { Map } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { loadGroupTable, GroupTableRow, loadForecastData } from "../utils/csvLoader";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

// Configure MapLibre worker for GitHub Pages with CSP support
// This ensures the worker loads correctly from the subdirectory
if (typeof window !== 'undefined') {
  const workerUrl = new URL(
    'maplibre-gl/dist/maplibre-gl-csp-worker.js',
    import.meta.url
  ).href;
  (maplibregl as any).workerUrl = workerUrl;
}

interface RegionMarker {
  id: string;
  name: string;
  subregion: string;
  city: string;
  groupId: number;
  lng: number;
  lat: number;
  predictedConsumption: number;
  trend: "up" | "down" | "stable";
}

interface MapSectionProps {
  selectedRegion: string;
  selectedCity: string;
  onMarkerClick: (
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
  ) => void;
}

// City to coordinates mapping (used to position markers)
const cityCoordinates: Record<string, { lat: number; lng: number }> = {
  "Joensuu": { lat: 62.60, lng: 29.76 },
  "Etelä-Savo": { lat: 61.50, lng: 27.50 },
  "Rovaniemi": { lat: 66.50, lng: 25.72 },
  "Lappi_Others": { lat: 68.00, lng: 26.00 },
  "Helsinki": { lat: 60.17, lng: 24.94 },
  "Turku": { lat: 60.45, lng: 22.27 },
  "Tampere": { lat: 61.50, lng: 23.76 },
  "Vaasa": { lat: 63.10, lng: 21.62 },
  "Pohjois-Karjala_Others": { lat: 63.00, lng: 30.00 },
  "Mikkeli": { lat: 61.69, lng: 27.27 },
  "Espoo": { lat: 60.21, lng: 24.66 },
  "Varsinais-Suomi_Others": { lat: 60.80, lng: 22.50 },
  "Pohjanmaa_Others": { lat: 63.50, lng: 22.00 },
  "Pirkanmaa_Others": { lat: 61.80, lng: 24.00 },
  // Additional cities from group table
  "Lappi": { lat: 67.00, lng: 26.00 },
  "Pohjois-Karjala": { lat: 63.00, lng: 30.00 },
  "Pohjois-Savo_Others": { lat: 63.50, lng: 28.00 },
  "Oulu": { lat: 65.01, lng: 25.47 },
  "Pohjois-Pohjanmaa": { lat: 65.00, lng: 25.50 },
  "Pohjois-Pohjanmaa_Others": { lat: 65.20, lng: 25.70 },
  "Lappeenranta": { lat: 61.06, lng: 28.19 },
  "Kanta-Häme": { lat: 61.00, lng: 24.50 },
  "Kanta-Häme_Others": { lat: 61.10, lng: 24.60 },
  "Lahti": { lat: 60.98, lng: 25.66 },
  "Päijät-Häme": { lat: 61.00, lng: 25.70 },
  "Päijät-Häme_Others": { lat: 61.10, lng: 25.80 },
  "Uusimaa_Others": { lat: 60.30, lng: 24.80 },
  "Vantaa": { lat: 60.29, lng: 25.04 },
  "Pori": { lat: 61.48, lng: 21.80 },
  "Etelä-Pohjanmaa": { lat: 62.79, lng: 23.13 },
  "Keski-Suomi": { lat: 62.24, lng: 25.75 },
  "Keski-Suomi_Others": { lat: 62.30, lng: 25.80 },
  "Jyväskylä": { lat: 62.24, lng: 25.75 },
  "Pirkanmaa": { lat: 61.50, lng: 23.80 },
  "Pohjanmaa": { lat: 63.10, lng: 22.00 },
};

// This will be populated from the group table CSV
const regionMap: Record<string, string> = {};

const getTrendIcon = (t: string) =>
  t === "up" ? "↑" : t === "down" ? "↓" : "→";

export function MapSection({
  selectedRegion,
  selectedCity,
  onMarkerClick,
}: MapSectionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<Map | null>(null);
  const loadedRef = useRef(false); // Track loaded state with ref to avoid stale closures
  const [loaded, setLoaded] = useState(false);
  const [groupTable, setGroupTable] = useState<GroupTableRow[]>([]);
  const [regionMapFromGroups, setRegionMapFromGroups] = useState<Record<string, string>>({});
  const [forecastDataCache, setForecastDataCache] = useState<Record<number, { hourly?: number; monthly?: number }>>({}); // Cache both hourly and monthly forecast values
  const [viewMode, setViewMode] = useState<'hourly' | 'monthly'>('hourly'); // Toggle between hourly and monthly view
  
  // Calculate average consumption for color palette
  const averageConsumption = React.useMemo(() => {
    const allConsumptions: { hourly: number[]; monthly: number[] } = { hourly: [], monthly: [] };
    
    // Collect all consumption values from forecast data cache
    (Object.values(forecastDataCache) as Array<{ hourly?: number; monthly?: number }>).forEach((data) => {
      if (data.hourly !== undefined) {
        allConsumptions.hourly.push(data.hourly);
      }
      if (data.monthly !== undefined) {
        allConsumptions.monthly.push(data.monthly);
      }
    });
    
    // If no forecast data, use estimated values from group table
    if (allConsumptions.hourly.length === 0 || allConsumptions.monthly.length === 0) {
      const dailyConsumptionMap: Record<string, number> = {
        "Low": 500,
        "Medium": 750,
        "High": 1200,
      };
      const monthlyConsumptionMap: Record<string, number> = {
        "Low": 15000,
        "Medium": 22500,
        "High": 36000,
      };
      
      groupTable.forEach(group => {
        const dailyEstimate = dailyConsumptionMap[group.consumptionLevel] || 700;
        const monthlyEstimate = monthlyConsumptionMap[group.consumptionLevel] || 21000;
        allConsumptions.hourly.push(dailyEstimate);
        allConsumptions.monthly.push(monthlyEstimate);
      });
    }
    
    const avgHourly = allConsumptions.hourly.length > 0
      ? allConsumptions.hourly.reduce((sum, val) => sum + val, 0) / allConsumptions.hourly.length
      : 700;
    const avgMonthly = allConsumptions.monthly.length > 0
      ? allConsumptions.monthly.reduce((sum, val) => sum + val, 0) / allConsumptions.monthly.length
      : 21000;
    
    return { hourly: avgHourly, monthly: avgMonthly };
  }, [forecastDataCache, groupTable]);
  
  // Filter states
  const [selectedCustomerType, setSelectedCustomerType] = useState<string>("all");
  const [selectedPriceType, setSelectedPriceType] = useState<string>("all");
  const [selectedConsumptionLevel, setSelectedConsumptionLevel] = useState<string>("all");
  
  // Location selection states
  const [selectedMapRegion, setSelectedMapRegion] = useState<string>("");
  const [selectedMapSubregion, setSelectedMapSubregion] = useState<string>("");
  const [selectedMapCity, setSelectedMapCity] = useState<string>("");

  const popupRef = useRef<maplibregl.Popup | null>(null);
  const topPopupRef = useRef<HTMLDivElement | null>(null);

  // Load group table CSV and build region mapping
  useEffect(() => {
    // In Vite, assets in src/assets need to be imported or accessed via public
    // Try both paths - adjust based on your setup
    loadGroupTable(new URL("../assets/data/groups-Table.csv", import.meta.url).href)
      .then((groups) => {
        setGroupTable(groups);
        
        // Build region map from group table
        const regionMapping: Record<string, string> = {};
        groups.forEach((group) => {
          if (group.subregion && group.region) {
            regionMapping[group.subregion] = group.region;
          }
        });
        setRegionMapFromGroups(regionMapping);
        
        console.log("Loaded group table:", groups.length, "groups");
        console.log("Region mapping:", regionMapping);
      })
      .catch((error) => {
        console.error("Failed to load group table, using fallback:", error);
        // Fallback to original region map
        setRegionMapFromGroups({
          "Pohjois-Karjala": "Eastern Finland",
          "Etelä-Savo": "Eastern Finland",
          Lappi: "Lapland",
          Uusimaa: "Southern Finland",
          "Varsinais-Suomi": "Southern Finland",
          Pohjanmaa: "Western Finland",
          Pirkanmaa: "Western Finland",
        });
      });
  }, []);

  // -------------------------------------------------------------
  // INITIALIZE PURE MAPLIBRE (SMOOTH ZOOM + TILT)
  // -------------------------------------------------------------
  useEffect(() => {
    if (!ref.current) {
      console.warn("Map container ref is null");
      return;
    }

    // Wait for container to have dimensions
    const checkAndInit = () => {
      if (!ref.current) return false;
      const rect = ref.current.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    };

    const initMap = () => {
      if (!checkAndInit() || map.current) return;

    const apiKey = "76bad9eb-0487-4e7b-bc13-4f01f6986346";

      // Use raster tiles for Stamen Toner (Stadia Maps format)
      const rasterTileUrl = `https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png?api_key=${apiKey}`;
      
      // Create a custom style with raster source
      const customStyle = {
        version: 8,
        sources: {
          'stamen-toner': {
            type: 'raster',
            tiles: [rasterTileUrl],
            tileSize: 256,
            attribution: '© Stadia Maps, © Stamen Design, © OpenMapTiles, © OpenStreetMap contributors'
          }
        },
        layers: [
          {
            id: 'stamen-toner-layer',
            type: 'raster',
            source: 'stamen-toner',
            minzoom: 0,
            maxzoom: 22
          }
        ]
      };

      console.log("Initializing map with container:", {
        width: ref.current!.offsetWidth,
        height: ref.current!.offsetHeight,
        tileUrl: rasterTileUrl
      });

      try {
    map.current = new maplibregl.Map({
          container: ref.current!,
          style: customStyle as any,
      center: [25, 62.5],
      zoom: 4.5,
      pitch: 0,
      bearing: 0,
      dragRotate: true,
      pitchWithRotate: true,
      interactive: true,
    });

    map.current.addControl(
      new maplibregl.NavigationControl({ visualizePitch: true })
    );

        const markAsLoaded = () => {
          if (!loadedRef.current) {
            loadedRef.current = true;
      setLoaded(true);
            console.log("Map marked as loaded");
            if (map.current) {
              map.current.resize();
              map.current.triggerRepaint();
              const canvas = ref.current?.querySelector('canvas');
              if (canvas) {
                console.log("Canvas dimensions:", canvas.width, "x", canvas.height);
              }
            }
          }
        };

        // Check if map is already loaded (in case load event fired before listener was attached)
        if (map.current.loaded()) {
          console.log("Map already loaded when listener attached");
          markAsLoaded();
        }

        map.current.on("load", () => {
          console.log("Map loaded successfully (load event)");
          markAsLoaded();
        });
        
        map.current.on("styledata", () => {
          console.log("Map style loaded");
          if (map.current) {
            map.current.resize();
            // Check if map is loaded after style loads
            if (map.current.loaded()) {
              console.log("Map loaded detected after styledata event");
              markAsLoaded();
            }
            // Log style sources to debug tile loading
            const style = map.current.getStyle();
            if (style && style.sources) {
              console.log("Style sources:", Object.keys(style.sources));
            }
          }
        });

        map.current.on("error", (e: any) => {
          console.error("Map error:", e);
          if (e.error) {
            console.error("Error details:", e.error);
            console.error("Error message:", e.error?.message);
            console.error("Error status:", e.error?.status);
          }
          // Log tile loading errors specifically
          if (e.error?.message?.includes('tile') || e.error?.message?.includes('Failed to load')) {
            console.error("Tile loading error detected!");
          }
        });

        map.current.on("sourcedata", (e: any) => {
          if (e.isSourceLoaded && e.sourceId) {
            console.log("Source loaded:", e.sourceId);
          }
          if (e.sourceId === 'stamen-toner' && e.isSourceLoaded) {
            console.log("Tile source 'stamen-toner' fully loaded!");
          }
        });

        map.current.on("sourcedataloading", (e: any) => {
          console.log("Source loading:", e.sourceId);
          if (e.sourceId === 'stamen-toner') {
            console.log("Tile source 'stamen-toner' is loading...");
          }
        });

        map.current.on("sourcedataerror", (e: any) => {
          console.error("Source data error:", e.sourceId, e.error);
          if (e.sourceId === 'stamen-toner') {
            console.error("CRITICAL: Tile source 'stamen-toner' failed to load!", e.error);
            console.error("Error details:", {
              sourceId: e.sourceId,
              error: e.error,
              tile: e.tile,
              status: e.tile?.status,
              message: e.error?.message
            });
          }
        });

        map.current.on("tileerror", (e: any) => {
          console.error("Tile error:", e);
          if (e.sourceId === 'stamen-toner') {
            console.error("CRITICAL: Tile failed to load for 'stamen-toner'!", {
              sourceId: e.sourceId,
              tile: e.tile,
              error: e.error,
              tileUrl: e.tile?.tileID,
              status: e.tile?.status
            });
          }
        });

        // Listen for when tiles are requested
        map.current.on("dataloading", (e: any) => {
          if (e.dataType === 'source' && e.sourceId === 'stamen-toner') {
            console.log("Tile data loading for stamen-toner");
          }
        });

        // Listen for when tiles finish loading
        map.current.on("data", (e: any) => {
          if (e.dataType === 'tile' && e.sourceId === 'stamen-toner') {
            if (e.isSourceLoaded) {
              console.log("Tile loaded successfully for stamen-toner");
            } else {
              console.log("Tile data event for stamen-toner:", {
                isSourceLoaded: e.isSourceLoaded,
                tile: e.tile
              });
            }
          }
          if (e.dataType === 'source' && e.isSourceLoaded) {
            console.log("Source data loaded:", e.sourceId);
          }
          if (e.dataType === 'source' && e.sourceId === 'stamen-toner') {
            console.log("stamen-toner data event:", {
              isSourceLoaded: e.isSourceLoaded,
              tile: e.tile
            });
          }
          if (e.dataType === 'source' && map.current?.loaded()) {
            console.log("Map loaded detected via data event");
            markAsLoaded();
          }
        });

        // Fallback: check periodically if map is loaded
        let checkCount = 0;
        const loadCheckInterval = setInterval(() => {
          checkCount++;
          const hasMap = !!map.current;
          const isLoaded = map.current?.loaded() || false;
          const hasCanvas = !!ref.current?.querySelector('canvas');
          const canvas = ref.current?.querySelector('canvas') as HTMLCanvasElement;
          const canvasSize = canvas ? { width: canvas.width, height: canvas.height } : null;
          
          console.log(`Periodic check #${checkCount}:`, {
            hasMap,
            isLoaded,
            hasCanvas,
            canvasSize,
            loadedRef: loadedRef.current
          });
          
          // If we have a canvas with dimensions, consider the map loaded (even if tiles aren't loading)
          // This allows markers to render even if tiles fail
          if (hasCanvas && canvasSize && canvasSize.width > 0 && canvasSize.height > 0 && !loadedRef.current) {
            console.log("Map loaded detected via canvas check (canvas has dimensions)");
            markAsLoaded();
            clearInterval(loadCheckInterval);
          } else if (isLoaded && !loadedRef.current) {
            console.log("Map loaded detected via periodic check (map.loaded() = true)");
            markAsLoaded();
            clearInterval(loadCheckInterval);
          } else if (hasCanvas && !loadedRef.current && checkCount >= 3) {
            // After a few checks, if canvas exists, mark as loaded even without dimensions
            // This handles cases where tiles fail but map is still functional
            console.log("Map loaded detected via canvas existence (fallback after checks)");
            markAsLoaded();
            clearInterval(loadCheckInterval);
          }
        }, 500);
        
        // Clear interval after 10 seconds
        setTimeout(() => {
          clearInterval(loadCheckInterval);
          // Final check
          const hasCanvas = !!ref.current?.querySelector('canvas');
          const canvas = ref.current?.querySelector('canvas') as HTMLCanvasElement;
          if (hasCanvas && canvas && canvas.width > 0 && canvas.height > 0 && !loadedRef.current) {
            console.log("Map loaded detected in final check (canvas exists)");
            markAsLoaded();
          } else if (map.current?.loaded() && !loadedRef.current) {
            console.log("Map loaded detected in final check (map.loaded())");
            markAsLoaded();
          } else {
            console.warn("Map never detected as loaded. Final state:", {
              hasMap: !!map.current,
              mapLoaded: map.current?.loaded(),
              hasCanvas,
              canvasSize: canvas ? { width: canvas.width, height: canvas.height } : null
            });
          }
        }, 10000);

        // ResizeObserver for container size changes
        const resizeObserver = new ResizeObserver(() => {
          if (map.current?.loaded()) {
            map.current.resize();
          }
        });
        resizeObserver.observe(ref.current!);

        // IntersectionObserver for visibility
        const intersectionObserver = new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              if (entry.isIntersecting && map.current?.loaded()) {
                setTimeout(() => {
                  map.current?.resize();
                }, 100);
              }
            });
          },
          { threshold: 0.01 }
        );
        intersectionObserver.observe(ref.current!);

    return () => {
          resizeObserver.disconnect();
          intersectionObserver.disconnect();
        };
      } catch (error) {
        console.error("Error creating map:", error);
      }
    };

    // Try to initialize immediately
    if (checkAndInit()) {
      setTimeout(initMap, 0);
    } else {
      // Wait and retry
      const retryInterval = setInterval(() => {
        if (checkAndInit()) {
          clearInterval(retryInterval);
          initMap();
        }
      }, 100);
      
      setTimeout(() => clearInterval(retryInterval), 5000);
    }

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []); // Only run once for map initialization

  // Generate markers from group table
  const generateMarkersFromGroups = (): RegionMarker[] => {
    if (groupTable.length === 0) return [];

    return groupTable.map((group) => {
      // Get coordinates for city
      const coords = cityCoordinates[group.city] || 
                     cityCoordinates[group.subregion] || 
                     { lat: 62.5, lng: 25.0 }; // Default to Finland center

      // Use real forecast data if available, otherwise fallback to estimated
      const forecastData = forecastDataCache[group.group_id];
      const forecastValue = forecastData ? (viewMode === 'hourly' ? forecastData.hourly : forecastData.monthly) : undefined;
      
      // Consumption map: daily values (kWh/day) for hourly, monthly values (kWh/month) for monthly
      const dailyConsumptionMap: Record<string, number> = {
        "Low": 500,
        "Medium": 750,
        "High": 1200,
      };
      const monthlyConsumptionMap: Record<string, number> = {
        "Low": 15000,   // 500 * 30 days
        "Medium": 22500, // 750 * 30 days
        "High": 36000,   // 1200 * 30 days
      };
      const consumptionMap = viewMode === 'hourly' ? dailyConsumptionMap : monthlyConsumptionMap;
      const estimatedConsumption = consumptionMap[group.consumptionLevel] || (viewMode === 'hourly' ? 700 : 21000);
      const predictedConsumption = forecastValue !== undefined ? forecastValue : estimatedConsumption;

      return {
        id: String(group.group_id),
        name: group.city || group.subregion,
        subregion: group.subregion,
        city: group.city,
        groupId: group.group_id,
        lng: coords.lng,
        lat: coords.lat,
        predictedConsumption: predictedConsumption,
        trend: "stable" as const,
      };
    });
  };

  // Filter markers based on selected criteria
  const getFilteredMarkers = (allMarkers: RegionMarker[]): RegionMarker[] => {
    return allMarkers.filter((marker) => {
      const groupInfo = groupTable.find(g => g.group_id === marker.groupId);
      if (!groupInfo) return false;

      // Filter by location (region, subregion, city)
      if (selectedMapRegion && groupInfo.region !== selectedMapRegion) {
        return false;
      }
      if (selectedMapSubregion && groupInfo.subregion !== selectedMapSubregion) {
        return false;
      }
      if (selectedMapCity && groupInfo.city !== selectedMapCity) {
        return false;
      }

      // Filter by customer type
      if (selectedCustomerType !== "all" && groupInfo.customerType !== selectedCustomerType) {
        return false;
      }

      // Filter by price type
      if (selectedPriceType !== "all" && groupInfo.priceType !== selectedPriceType) {
        return false;
      }

      // Filter by consumption level
      if (selectedConsumptionLevel !== "all" && groupInfo.consumptionLevel !== selectedConsumptionLevel) {
        return false;
      }

      return true;
    });
  };

  // Load forecast data for filtered markers
  useEffect(() => {
    if (groupTable.length === 0) return;

    // Get filtered groups based on current selections
    const filteredGroups = groupTable.filter((g) => {
      if (selectedMapRegion && g.region !== selectedMapRegion) return false;
      if (selectedMapSubregion && g.subregion !== selectedMapSubregion) return false;
      if (selectedMapCity && g.city !== selectedMapCity) return false;
      if (selectedCustomerType !== "all" && g.customerType !== selectedCustomerType) return false;
      if (selectedPriceType !== "all" && g.priceType !== selectedPriceType) return false;
      if (selectedConsumptionLevel !== "all" && g.consumptionLevel !== selectedConsumptionLevel) return false;
      return true;
    });

    // Load forecast data for each filtered group (both hourly and monthly)
    const loadForecasts = async () => {
      const newCache: Record<number, { hourly?: number; monthly?: number }> = {};
      
      await Promise.all(
        filteredGroups.map(async (group) => {
          try {
            // Load both hourly (48h) and monthly (12m) forecasts
            const [hourlyData, monthlyData] = await Promise.all([
              loadForecastData(
                new URL('../assets/data/forecast_48h.csv', import.meta.url).href,
                group.group_id,
                false
              ),
              loadForecastData(
                new URL('../assets/data/forecast_12m.csv', import.meta.url).href,
                group.group_id,
                true
              )
            ]);
            
            const cacheEntry: { hourly?: number; monthly?: number } = {};
            
            if (hourlyData && hourlyData.length > 0) {
              // Use the first hour's consumption value
              cacheEntry.hourly = hourlyData[0].consumption;
            }
            
            if (monthlyData && monthlyData.length > 0) {
              // Use the first month's consumption value
              cacheEntry.monthly = monthlyData[0].consumption;
            }
            
            if (cacheEntry.hourly || cacheEntry.monthly) {
              newCache[group.group_id] = cacheEntry;
            }
          } catch (error) {
            console.warn(`Failed to load forecast for group ${group.group_id}:`, error);
          }
        })
      );
      
      setForecastDataCache(newCache);
    };

    loadForecasts();
  }, [groupTable, selectedMapRegion, selectedMapSubregion, selectedMapCity, selectedCustomerType, selectedPriceType, selectedConsumptionLevel, viewMode]);

  // Create markers when map is loaded AND group table is loaded
  useEffect(() => {
    if (!loaded || !map.current) {
      console.log("Markers: waiting for map to load", { loaded, hasMap: !!map.current });
      return;
    }
    
    // Wait for group table to load
    if (groupTable.length === 0) {
      console.log("Markers: waiting for group table to load");
      return;
    }

    console.log("Markers: Creating markers", { groupTableLength: groupTable.length, mapLoaded: map.current.loaded() });

    // Clear existing markers first
    const existingMarkers = document.querySelectorAll('.maplibregl-marker');
    console.log(`Markers: Clearing ${existingMarkers.length} existing markers`);
    existingMarkers.forEach(m => m.remove());

    // Generate all markers from group table
    const allMarkers = generateMarkersFromGroups();
    const filteredMarkers = getFilteredMarkers(allMarkers);

    console.log(`Markers: Displaying ${filteredMarkers.length} of ${allMarkers.length} markers`);

    // Store allMarkers in a variable accessible to event handlers
    const markersForHandlers = allMarkers;

    filteredMarkers.forEach((m) => {
      // Get group info from group table by group_id
      const groupInfo = groupTable.find(g => g.group_id === m.groupId);
        
        // Use group table data for region and city, fallback to marker data
        const region = groupInfo?.region || regionMapFromGroups[m.subregion] || regionMap[m.subregion] || "Unknown";
        const city = groupInfo?.city || m.city;
        const subregion = groupInfo?.subregion || m.subregion;
        
        const isRegionSelected = selectedRegion && region === selectedRegion;

        // -----------------------------
        // MARKER DOM ELEMENT
        // -----------------------------
        // Don't set position styles on the root element - MapLibre GL handles positioning
        const el = document.createElement("div");
        el.style.width = "100px";
        el.style.height = "100px";
        el.style.cursor = "pointer";
        el.style.pointerEvents = "auto";
        el.style.touchAction = "none";
        el.className = "custom-marker-clickable";
        
        // Make sure the element can receive events
        el.setAttribute('data-marker-id', m.id);

        // Calculate intensity based on average consumption
        const avgConsumption = viewMode === 'hourly' ? averageConsumption.hourly : averageConsumption.monthly;
        const belowAvgThreshold = avgConsumption * 0.8;
        const aboveAvgThreshold = avgConsumption * 1.2;
        const veryHighThreshold = avgConsumption * 1.5;
        
        let intensity: number;
        if (m.predictedConsumption < belowAvgThreshold) {
          intensity = 2; // Below average
        } else if (m.predictedConsumption <= aboveAvgThreshold) {
          intensity = 3; // Around average
        } else if (m.predictedConsumption <= veryHighThreshold) {
          intensity = 4; // Moderately above average
        } else {
          intensity = 5; // Significantly above average
        }

        // Create inner content wrapper with relative positioning
        const contentWrapper = document.createElement("div");
        contentWrapper.style.position = "relative";
        contentWrapper.style.width = "100%";
        contentWrapper.style.height = "100%";
        contentWrapper.style.pointerEvents = "none";
        
        // User-friendly color palette based on average consumption
        // 3-level system: Below average (Green), Around average (Blue), Above average (Orange/Red)
        // Thresholds are calculated dynamically based on actual data averages
        const getColorScheme = (consumption: number, isSelected: boolean) => {
          // Use average consumption as the middle threshold
          const avgConsumption = viewMode === 'hourly' ? averageConsumption.hourly : averageConsumption.monthly;
          
          // Define thresholds: below average (0.8x avg), around average (0.8x-1.2x avg), above average (>1.2x avg)
          const belowAvgThreshold = avgConsumption * 0.8;
          const aboveAvgThreshold = avgConsumption * 1.2;
          
          if (consumption < belowAvgThreshold) {
            // Below average consumption - Green
            return {
              primary: isSelected ? "#10B981" : "#34D399", // Emerald green
              border: isSelected ? "#059669" : "#10B981", // Darker green
              glow: "rgba(16, 185, 129, 0.4)", // Green glow
              labelBg: "#ECFDF5", // Light green background
              labelText: "#065F46" // Dark green text
            };
          } else if (consumption <= aboveAvgThreshold) {
            // Around average consumption - Blue
            return {
              primary: isSelected ? "#3B82F6" : "#60A5FA", // Blue
              border: isSelected ? "#2563EB" : "#3B82F6", // Darker blue
              glow: "rgba(59, 130, 246, 0.4)", // Blue glow
              labelBg: "#EFF6FF", // Light blue background
              labelText: "#1E40AF" // Dark blue text
            };
          } else {
            // Above average consumption - Orange/Red
            // Use orange for moderately above, red for significantly above
            if (consumption <= avgConsumption * 1.5) {
              // Moderately above average - Orange
              return {
                primary: isSelected ? "#F59E0B" : "#FBBF24", // Amber/Orange
                border: isSelected ? "#D97706" : "#F59E0B", // Darker orange
                glow: "rgba(245, 158, 11, 0.4)", // Orange glow
                labelBg: "#FFFBEB", // Light amber background
                labelText: "#92400E" // Dark amber text
              };
            } else {
              // Significantly above average - Red
              return {
                primary: isSelected ? "#EF4444" : "#F87171", // Red
                border: isSelected ? "#DC2626" : "#EF4444", // Darker red
                glow: "rgba(239, 68, 68, 0.4)", // Red glow
                labelBg: "#FEF2F2", // Light red background
                labelText: "#991B1B" // Dark red text
              };
            }
          }
        };
        
        const colors = getColorScheme(m.predictedConsumption, !!isRegionSelected);
        
        contentWrapper.innerHTML =
          [...Array(intensity)]
            .map((_, i) => {
            const size = 32 + i * 12;
              const opacity = 0.3 - i * 0.06;
              return `
            <div style="
              position:absolute;
              top:50%; left:50%;
              transform:translate(-50%,-50%);
              width:${size}px; height:${size}px;
              border-radius:50%;
              border:2px solid ${colors.border};
              background:${colors.primary};
              opacity:${opacity};
              pointer-events:none;
              box-shadow:0 0 ${8 + i * 2}px ${colors.glow};
            "></div>
          `;
            })
            .join("") +
          `
          <div style="
            position:absolute;
            top:50%; left:50%;
            transform:translate(-50%,-50%);
            width:44px; height:44px;
            border-radius:50%;
            border:3px solid ${colors.border};
            background:linear-gradient(135deg, ${colors.primary} 0%, ${colors.border} 100%);
            box-shadow:0 4px 16px rgba(0,0,0,0.3), 0 0 0 2px rgba(255,255,255,0.1), inset 0 2px 4px rgba(255,255,255,0.3);
            pointer-events:none;
          ">
            <div style="
              position:absolute;
              top:50%; left:50%;
              transform:translate(-50%,-50%);
              width:18px; height:18px;
              border-radius:50%;
              background:${colors.border};
              box-shadow:0 2px 6px rgba(0,0,0,0.4), inset 0 1px 3px rgba(255,255,255,0.4);
            "></div>
          </div>
          <div style="
            position:absolute;
            top:78px; left:50%;
            transform:translateX(-50%);
            background:${colors.labelBg};
            font-size:11px;
            font-weight:600;
            padding:6px 10px;
            border-radius:6px;
            border:1.5px solid ${colors.border};
            color:${colors.labelText};
            white-space:nowrap;
            pointer-events:none;
            box-shadow:0 3px 10px rgba(0,0,0,0.2), 0 1px 3px rgba(0,0,0,0.1);
            letter-spacing:0.3px;
          ">
            ${m.predictedConsumption.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} ${viewMode === 'hourly' ? 'kWh/day' : 'kWh/month'}
            </div>
          `;

        el.appendChild(contentWrapper);

        // Define click handler BEFORE creating marker
        const handleClick = (e: Event) => {
              e.stopPropagation();
          e.preventDefault();
          e.stopImmediatePropagation();
          
          console.log('Marker clicked:', m.name, 'Group ID:', m.groupId);
          
          // Use group table data for accurate region and city
          const finalRegion = groupInfo?.region || regionMapFromGroups[m.subregion] || regionMap[m.subregion] || "Unknown";
          const finalCity = groupInfo?.city || m.city;
          const finalSubregion = groupInfo?.subregion || m.subregion;
          
          // Check if the clicked marker's group matches current filters
          // If filters are active, find a group that matches both city and filters
          let groupIdToUse = m.groupId;
          let groupToUse = groupInfo;
          
          // If filters are active, try to find a better matching group
          if (selectedCustomerType !== "all" || selectedPriceType !== "all" || selectedConsumptionLevel !== "all") {
            const matchingGroup = groupTable.find(g => 
              g.city === finalCity &&
              (selectedCustomerType === "all" || g.customerType === selectedCustomerType) &&
              (selectedPriceType === "all" || g.priceType === selectedPriceType) &&
              (selectedConsumptionLevel === "all" || g.consumptionLevel === selectedConsumptionLevel)
            );
            
            if (matchingGroup) {
              groupIdToUse = matchingGroup.group_id;
              groupToUse = matchingGroup;
            } else if (groupInfo) {
              // If no perfect match but the clicked group matches, use it
              const matchesFilters = 
                (selectedCustomerType === "all" || groupInfo.customerType === selectedCustomerType) &&
                (selectedPriceType === "all" || groupInfo.priceType === selectedPriceType) &&
                (selectedConsumptionLevel === "all" || groupInfo.consumptionLevel === selectedConsumptionLevel);
              
              if (!matchesFilters) {
                // Clicked group doesn't match filters, but we'll use it anyway
                console.warn('Clicked marker group does not match active filters');
              }
            }
          }

          // Remove existing popup
          popupRef.current?.remove();

          // Don't call onMarkerClick here - only show popup
          // Graph will only open when a specific group ID is clicked in the popup

          // Find all markers/groups for this city
          const allCityGroups = groupTable.filter(g => g.city === finalCity);
          
          // Update top popup element if it exists
          if (topPopupRef.current) {
            if (allCityGroups.length > 1) {
              // Show all options for this city
              const optionsHtml = allCityGroups.map((group, idx) => {
                const groupForecastData = forecastDataCache[group.group_id];
                const forecastValue = groupForecastData ? (viewMode === 'hourly' ? groupForecastData.hourly : groupForecastData.monthly) : undefined;
                
                // Consumption map: daily values (kWh/day) for hourly, monthly values (kWh/month) for monthly
                const dailyConsumptionMap: Record<string, number> = {
                  "Low": 500,
                  "Medium": 750,
                  "High": 1200,
                };
                const monthlyConsumptionMap: Record<string, number> = {
                  "Low": 15000,   // 500 * 30 days
                  "Medium": 22500, // 750 * 30 days
                  "High": 36000,   // 1200 * 30 days
                };
                const consumptionMap = viewMode === 'hourly' ? dailyConsumptionMap : monthlyConsumptionMap;
                const estimatedConsumption = consumptionMap[group.consumptionLevel] || (viewMode === 'hourly' ? 700 : 21000);
                const predictedConsumption = forecastValue !== undefined ? forecastValue : estimatedConsumption;
                const unit = viewMode === 'hourly' ? 'kWh/day' : 'kWh/month';
                
                return `
                  <div 
                    onclick="
                      (function() {
                        const event = new CustomEvent('selectGroup', { detail: ${group.group_id}, bubbles: true });
                        window.dispatchEvent(event);
                      })();
                    "
                    style="
                      padding:10px 12px;
                      margin-bottom:8px;
                      background:#1e293b;
                      border:1px solid #334155;
                      border-radius:6px;
                      cursor:pointer;
                      transition:background 0.2s;
                    "
                    onmouseover="this.style.background='#334155'"
                    onmouseout="this.style.background='#1e293b'"
                  >
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                      <b style="font-size:14px;color:white;">Group ID: ${group.group_id}</b>
                      <span style="font-size:16px;color:#3B82F6;font-weight:bold;">${predictedConsumption.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} ${unit}</span>
                    </div>
                    <div style="color:#94a3b8;font-size:11px;">
                      <span style="margin-right:12px;"><strong>Customer:</strong> ${group.customerType}</span>
                      <span style="margin-right:12px;"><strong>Price:</strong> ${group.priceType}</span>
                      <span><strong>Level:</strong> ${group.consumptionLevel}</span>
                    </div>
                  </div>
                `;
              }).join('');
              
              topPopupRef.current.innerHTML = `
                <div style="padding:12px 16px;min-width:300px;max-width:500px;font-size:14px;color:white;background:#0f172a;border-radius:8px;border:2px solid #334155;box-shadow:0 4px 12px rgba(0,0,0,0.3);">
                  <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:12px;">
                    <div>
                      <b style="font-size:18px;">${finalCity || m.name}</b>
                      <div style="color:#cbd5e1;font-size:12px;margin-top:4px;">
                        <strong>Region:</strong> ${finalRegion} • <strong>Subregion:</strong> ${finalSubregion}
                      </div>
                      <div style="color:#94a3b8;font-size:11px;margin-top:6px;">
                        ${allCityGroups.length} options available
                      </div>
                    </div>
                    <button onclick="this.closest('.top-popup-container').style.display='none'" style="background:none;border:none;color:#cbd5e1;cursor:pointer;font-size:20px;line-height:1;padding:0;width:24px;height:24px;display:flex;align-items:center;justify-content:center;">×</button>
                  </div>
                  <div style="max-height:400px;overflow-y:auto;margin-top:8px;">
                    ${optionsHtml}
                  </div>
                </div>
              `;
              } else {
              // Single marker - show simple popup with clickable group ID
              const groupForecastData = forecastDataCache[m.groupId];
              const forecastValue = groupForecastData ? (viewMode === 'hourly' ? groupForecastData.hourly : groupForecastData.monthly) : undefined;
              
              // Consumption map: daily values (kWh/day) for hourly, monthly values (kWh/month) for monthly
              const dailyConsumptionMap: Record<string, number> = {
                "Low": 500,
                "Medium": 750,
                "High": 1200,
              };
              const monthlyConsumptionMap: Record<string, number> = {
                "Low": 15000,   // 500 * 30 days
                "Medium": 22500, // 750 * 30 days
                "High": 36000,   // 1200 * 30 days
              };
              const consumptionMap = viewMode === 'hourly' ? dailyConsumptionMap : monthlyConsumptionMap;
              const estimatedConsumption = consumptionMap[groupInfo?.consumptionLevel || 'Medium'] || (viewMode === 'hourly' ? 700 : 21000);
              const displayValue = forecastValue !== undefined ? forecastValue : estimatedConsumption;
              const unit = viewMode === 'hourly' ? 'kWh/day' : 'kWh/month';
              const viewLabel = viewMode === 'hourly' ? 'Hourly' : 'Monthly';
              
              // Use the group's actual attributes as filters (not the filter state)
              const filters = groupInfo ? {
                customerType: groupInfo.customerType || "",
                priceType: groupInfo.priceType || "",
                consumptionLevel: groupInfo.consumptionLevel || "",
              } : undefined;
              
              topPopupRef.current.innerHTML = `
                <div style="padding:12px 16px;min-width:200px;font-size:14px;color:white;background:#0f172a;border-radius:8px;border:2px solid #334155;box-shadow:0 4px 12px rgba(0,0,0,0.3);">
                  <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:8px;">
                    <b style="font-size:18px;">${finalCity || m.name}</b>
                    <button onclick="this.closest('.top-popup-container').style.display='none'" style="background:none;border:none;color:#cbd5e1;cursor:pointer;font-size:20px;line-height:1;padding:0;width:24px;height:24px;display:flex;align-items:center;justify-content:center;">×</button>
                  </div>
                <span style="color:#cbd5e1;font-size:12px;margin-top:4px;display:block;">
                  <strong>Region:</strong> ${finalRegion}<br/>
                  <strong>Subregion:</strong> ${finalSubregion}<br/>
                  <strong>Group ID:</strong> <span 
                    onclick="
                      (function() {
                        const event = new CustomEvent('selectGroup', { detail: ${m.groupId}, bubbles: true });
                        window.dispatchEvent(event);
                      })();
                    "
                    style="
                      color:#3B82F6;
                      cursor:pointer;
                      text-decoration:underline;
                      font-weight:bold;
                    "
                    onmouseover="this.style.color='#60A5FA'"
                    onmouseout="this.style.color='#3B82F6'"
                  >${m.groupId}</span>
                </span>
                ${groupInfo ? `
                   <div style="color:#94a3b8;font-size:11px;margin-top:8px;padding-top:8px;border-top:1px solid #334155;">
                  <strong>Customer:</strong> ${groupInfo.customerType}<br/>
                  <strong>Price Type:</strong> ${groupInfo.priceType}<br/>
                  <strong>Consumption:</strong> ${groupInfo.consumptionLevel}
                </div>
                ` : ''}
                   <hr style="border-color:#334155;margin:10px 0;" />
                 <span style="color:#cbd5e1;font-size:12px;">${viewLabel} prediction</span>
                   <div style="margin-top:6px;">
                     <b style="font-size:16px;">${displayValue.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} ${unit}</b>
                     <span style="margin-left:8px;color:#94a3b8;">${getTrendIcon(m.trend)}</span>
                </div>
              </div>
               `;
               
               // Set up event listener for single group selection
               // Remove any existing listener first
               const existingSingleHandler = (window as any).__singleGroupSelectHandler;
               if (existingSingleHandler) {
                 window.removeEventListener('selectGroup', existingSingleHandler as EventListener);
               }
               
               const handleSingleGroupSelect = (event: CustomEvent) => {
                 console.log('Single group select event received:', event.detail, 'Marker groupId:', m.groupId);
                 const selectedGroupId = event.detail;
                 if (selectedGroupId === m.groupId) {
                   console.log('Opening graph for group:', selectedGroupId);
                   // Call onMarkerClick to open the graph
                   onMarkerClick(
                     finalRegion,
                     finalSubregion,
                     finalCity,
                     m.groupId,
                     filters,
                     viewMode
                   );
                   
                   // Scroll to graph section
                   setTimeout(() => {
                     const graphSection = document.getElementById('graph-section');
                     if (graphSection) {
                       graphSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                     }
                   }, 300);
                 }
               };
               
               // Store handler reference
               (window as any).__singleGroupSelectHandler = handleSingleGroupSelect;
               window.addEventListener('selectGroup', handleSingleGroupSelect as EventListener);
             }
            topPopupRef.current.style.display = 'block';
          }
          
          // Set up event listener for group selection (only if multiple groups)
          if (allCityGroups.length > 1) {
            // Remove any existing listener first
            const existingHandler = (window as any).__groupSelectHandler;
            if (existingHandler) {
              window.removeEventListener('selectGroup', existingHandler as EventListener);
            }
            
            const handleGroupSelect = (event: CustomEvent) => {
              console.log('Group select event received:', event.detail);
              const selectedGroupId = event.detail;
              const selectedGroup = groupTable.find(g => g.group_id === selectedGroupId);
              
              console.log('Selected group found:', selectedGroup);
              
              if (selectedGroup) {
                // Find the matching marker
                const selectedMarker = markersForHandlers.find(marker => marker.groupId === selectedGroupId);
                
                if (selectedMarker) {
                  // Trigger the same logic as clicking the marker
                  const finalRegion = selectedGroup.region || regionMapFromGroups[selectedGroup.subregion] || regionMap[selectedGroup.subregion] || "Unknown";
                  const finalCity = selectedGroup.city;
                  const finalSubregion = selectedGroup.subregion;
                  
                  // Use the group's actual attributes as filters (not the filter state)
                  const filters = {
                    customerType: selectedGroup.customerType || "",
                    priceType: selectedGroup.priceType || "",
                    consumptionLevel: selectedGroup.consumptionLevel || "",
                  };

                  onMarkerClick(
                    finalRegion,
                    finalSubregion,
                    finalCity, 
                    selectedGroupId,
                    filters,
                    viewMode
                  );

                  // Scroll to graph section after a short delay to ensure state updates
                  setTimeout(() => {
                    const graphSection = document.getElementById('graph-section');
                    if (graphSection) {
                      graphSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                  }, 300);
                  
                  // Update popup to show selected group
                  if (topPopupRef.current) {
                    const groupForecastData = forecastDataCache[selectedGroupId];
                    const forecastValue = groupForecastData ? (viewMode === 'hourly' ? groupForecastData.hourly : groupForecastData.monthly) : undefined;
                    
                    // Consumption map: daily values (kWh/day) for hourly, monthly values (kWh/month) for monthly
                    const dailyConsumptionMap: Record<string, number> = {
                      "Low": 500,
                      "Medium": 750,
                      "High": 1200,
                    };
                    const monthlyConsumptionMap: Record<string, number> = {
                      "Low": 15000,   // 500 * 30 days
                      "Medium": 22500, // 750 * 30 days
                      "High": 36000,   // 1200 * 30 days
                    };
                    const consumptionMap = viewMode === 'hourly' ? dailyConsumptionMap : monthlyConsumptionMap;
                    const estimatedConsumption = consumptionMap[selectedGroup.consumptionLevel] || (viewMode === 'hourly' ? 700 : 21000);
                    const predictedConsumption = forecastValue !== undefined ? forecastValue : estimatedConsumption;
                    const unit = viewMode === 'hourly' ? 'kWh/day' : 'kWh/month';
                    const viewLabel = viewMode === 'hourly' ? 'Hourly' : 'Monthly';
                    
                    topPopupRef.current.innerHTML = `
                      <div style="padding:12px 16px;min-width:200px;font-size:14px;color:white;background:#0f172a;border-radius:8px;border:2px solid #334155;box-shadow:0 4px 12px rgba(0,0,0,0.3);">
                        <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:8px;">
                          <b style="font-size:18px;">${finalCity}</b>
                          <button onclick="this.closest('.top-popup-container').style.display='none'" style="background:none;border:none;color:#cbd5e1;cursor:pointer;font-size:20px;line-height:1;padding:0;width:24px;height:24px;display:flex;align-items:center;justify-content:center;">×</button>
                        </div>
                        <span style="color:#cbd5e1;font-size:12px;margin-top:4px;display:block;">
                          <strong>Region:</strong> ${finalRegion}<br/>
                          <strong>Subregion:</strong> ${finalSubregion}<br/>
                          <strong>Group ID:</strong> ${selectedGroupId}
                        </span>
                        <div style="color:#94a3b8;font-size:11px;margin-top:8px;padding-top:8px;border-top:1px solid #334155;">
                          <strong>Customer:</strong> ${selectedGroup.customerType}<br/>
                          <strong>Price Type:</strong> ${selectedGroup.priceType}<br/>
                          <strong>Consumption:</strong> ${selectedGroup.consumptionLevel}
                        </div>
                        <hr style="border-color:#334155;margin:10px 0;" />
                        <span style="color:#cbd5e1;font-size:12px;">${viewLabel} prediction</span>
                        <div style="margin-top:6px;">
                          <b style="font-size:16px;">${predictedConsumption.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} ${unit}</b>
                        </div>
                      </div>
                    `;
                  }
                }
              }
            };
            
            // Store handler reference to allow cleanup
            (window as any).__groupSelectHandler = handleGroupSelect;
            window.addEventListener('selectGroup', handleGroupSelect as EventListener);
          }
        };
        
        // Attach event listeners BEFORE adding to map
        // Use capture phase and non-passive to ensure we catch the event
        el.addEventListener('click', handleClick, { capture: true, passive: false });
        el.addEventListener('mousedown', handleClick, { capture: true, passive: false });
        
        // Also handle touch events for mobile
        const handleTouch = (e: TouchEvent) => {
          e.stopPropagation();
          e.preventDefault();
          e.stopImmediatePropagation();
          // Trigger the click handler
          const syntheticEvent = new MouseEvent('click', {
            bubbles: false,
            cancelable: true,
            view: window
          });
          handleClick(syntheticEvent);
        };
        el.addEventListener('touchend', handleTouch, { capture: true, passive: false });
        
        // Create marker AFTER setting up event listeners
        // Use 'center' anchor so the marker center point is at the lat/lng
        try {
        const marker = new maplibregl.Marker({
          element: el,
          anchor: 'center'
        })
          .setLngLat([m.lng, m.lat])
          .addTo(map.current!);
        
        // Store marker reference for potential cleanup
        (el as any)._marker = marker;
          console.log(`Marker added for ${m.name} at [${m.lng}, ${m.lat}]`);
        } catch (error) {
          console.error(`Error adding marker for ${m.name}:`, error);
        }
      });
  }, [loaded, groupTable, regionMapFromGroups, selectedRegion, selectedCustomerType, selectedPriceType, selectedConsumptionLevel, selectedMapRegion, selectedMapSubregion, selectedMapCity, forecastDataCache, viewMode, onMarkerClick]);

  // -------------------------------------------------------------
  // FLY TO SELECTED CITY
  // -------------------------------------------------------------
  useEffect(() => {
    if (!loaded || !map.current || !selectedCity || groupTable.length === 0) return;

    // Find the marker for the selected city
    const allMarkers = generateMarkersFromGroups();
    const m = allMarkers.find((x) => x.city === selectedCity);
    if (!m) return;

    map.current.flyTo({
      center: [m.lng, m.lat],
      zoom: 7,
      pitch: 0,
      bearing: 0,
      speed: 1.3,
      curve: 1.2,
      essential: true,
    });
  }, [selectedCity, loaded, groupTable]);

  // When filters change and a city is selected via dropdown, update the group ID to match filters
  useEffect(() => {
    // Only update if city was selected via dropdown (selectedMapCity) and not via marker click
    if (!selectedMapCity || groupTable.length === 0) return;
    
    // Find a group that matches the city and all active filters
    const matchingGroup = groupTable.find(g => 
      g.city === selectedMapCity &&
      (selectedCustomerType === "all" || g.customerType === selectedCustomerType) &&
      (selectedPriceType === "all" || g.priceType === selectedPriceType) &&
      (selectedConsumptionLevel === "all" || g.consumptionLevel === selectedConsumptionLevel)
    );
    
    if (matchingGroup) {
      // Only pass filters if at least one is active
      const hasActiveFilters = selectedCustomerType !== "all" || selectedPriceType !== "all" || selectedConsumptionLevel !== "all";
      const filters = hasActiveFilters ? {
        customerType: selectedCustomerType !== "all" ? selectedCustomerType : "",
        priceType: selectedPriceType !== "all" ? selectedPriceType : "",
        consumptionLevel: selectedConsumptionLevel !== "all" ? selectedConsumptionLevel : "",
      } : undefined;

      // Update the selection with the matching group ID
      onMarkerClick(
        matchingGroup.region || selectedMapRegion,
        matchingGroup.subregion || selectedMapSubregion,
        selectedMapCity,
        matchingGroup.group_id,
        filters
      );
    }
  }, [selectedCustomerType, selectedPriceType, selectedConsumptionLevel, selectedMapCity, selectedMapRegion, selectedMapSubregion, groupTable, onMarkerClick]);

  // Get available options based on current selections (smart filtering)
  const getAvailableOptions = () => {
    // Start with all groups
    let filteredGroups = [...groupTable];

    // Filter by location if selected
    if (selectedMapRegion) {
      filteredGroups = filteredGroups.filter(g => g.region === selectedMapRegion);
    }
    if (selectedMapSubregion) {
      filteredGroups = filteredGroups.filter(g => g.subregion === selectedMapSubregion);
    }
    if (selectedMapCity) {
      filteredGroups = filteredGroups.filter(g => g.city === selectedMapCity);
    }

    // Filter by other active filters (excluding the one we're calculating)
    if (selectedCustomerType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.customerType === selectedCustomerType);
    }
    if (selectedPriceType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.priceType === selectedPriceType);
    }
    if (selectedConsumptionLevel !== "all") {
      filteredGroups = filteredGroups.filter(g => g.consumptionLevel === selectedConsumptionLevel);
    }

    return filteredGroups;
  };

  // Get available customer types based on current selections
  const getAvailableCustomerTypes = () => {
    let filteredGroups = [...groupTable];

    // Filter by location
    if (selectedMapRegion) {
      filteredGroups = filteredGroups.filter(g => g.region === selectedMapRegion);
    }
    if (selectedMapSubregion) {
      filteredGroups = filteredGroups.filter(g => g.subregion === selectedMapSubregion);
    }
    if (selectedMapCity) {
      filteredGroups = filteredGroups.filter(g => g.city === selectedMapCity);
    }

    // Filter by other active filters (but not customer type itself)
    if (selectedPriceType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.priceType === selectedPriceType);
    }
    if (selectedConsumptionLevel !== "all") {
      filteredGroups = filteredGroups.filter(g => g.consumptionLevel === selectedConsumptionLevel);
    }

    return Array.from(new Set(filteredGroups.map(g => g.customerType).filter(Boolean))).sort();
  };

  // Get available price types based on current selections
  const getAvailablePriceTypes = () => {
    let filteredGroups = [...groupTable];

    // Filter by location
    if (selectedMapRegion) {
      filteredGroups = filteredGroups.filter(g => g.region === selectedMapRegion);
    }
    if (selectedMapSubregion) {
      filteredGroups = filteredGroups.filter(g => g.subregion === selectedMapSubregion);
    }
    if (selectedMapCity) {
      filteredGroups = filteredGroups.filter(g => g.city === selectedMapCity);
    }

    // Filter by other active filters (but not price type itself)
    if (selectedCustomerType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.customerType === selectedCustomerType);
    }
    if (selectedConsumptionLevel !== "all") {
      filteredGroups = filteredGroups.filter(g => g.consumptionLevel === selectedConsumptionLevel);
    }

    return Array.from(new Set(filteredGroups.map(g => g.priceType).filter(Boolean))).sort();
  };

  // Get available consumption levels based on current selections
  const getAvailableConsumptionLevels = () => {
    let filteredGroups = [...groupTable];

    // Filter by location
    if (selectedMapRegion) {
      filteredGroups = filteredGroups.filter(g => g.region === selectedMapRegion);
    }
    if (selectedMapSubregion) {
      filteredGroups = filteredGroups.filter(g => g.subregion === selectedMapSubregion);
    }
    if (selectedMapCity) {
      filteredGroups = filteredGroups.filter(g => g.city === selectedMapCity);
    }

    // Filter by other active filters (but not consumption level itself)
    if (selectedCustomerType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.customerType === selectedCustomerType);
    }
    if (selectedPriceType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.priceType === selectedPriceType);
    }

    return Array.from(new Set(filteredGroups.map(g => g.consumptionLevel).filter(Boolean))).sort();
  };

  // Get available regions based on current filters
  const getAvailableRegions = () => {
    let filteredGroups = [...groupTable];

    // Filter by active filters
    if (selectedCustomerType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.customerType === selectedCustomerType);
    }
    if (selectedPriceType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.priceType === selectedPriceType);
    }
    if (selectedConsumptionLevel !== "all") {
      filteredGroups = filteredGroups.filter(g => g.consumptionLevel === selectedConsumptionLevel);
    }

    return Array.from(new Set(filteredGroups.map(g => g.region).filter(Boolean))).sort();
  };

  // Get available subregions based on current selections
  const getAvailableSubregions = () => {
    let filteredGroups = [...groupTable];

    // Filter by region if selected
    if (selectedMapRegion) {
      filteredGroups = filteredGroups.filter(g => g.region === selectedMapRegion);
    }

    // Filter by active filters
    if (selectedCustomerType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.customerType === selectedCustomerType);
    }
    if (selectedPriceType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.priceType === selectedPriceType);
    }
    if (selectedConsumptionLevel !== "all") {
      filteredGroups = filteredGroups.filter(g => g.consumptionLevel === selectedConsumptionLevel);
    }

    return Array.from(new Set(filteredGroups.map(g => g.subregion).filter(Boolean))).sort();
  };

  // Get available cities based on current selections
  const getAvailableCities = () => {
    let filteredGroups = [...groupTable];

    // Filter by location
    if (selectedMapRegion) {
      filteredGroups = filteredGroups.filter(g => g.region === selectedMapRegion);
    }
    if (selectedMapSubregion) {
      filteredGroups = filteredGroups.filter(g => g.subregion === selectedMapSubregion);
    }

    // Filter by active filters
    if (selectedCustomerType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.customerType === selectedCustomerType);
    }
    if (selectedPriceType !== "all") {
      filteredGroups = filteredGroups.filter(g => g.priceType === selectedPriceType);
    }
    if (selectedConsumptionLevel !== "all") {
      filteredGroups = filteredGroups.filter(g => g.consumptionLevel === selectedConsumptionLevel);
    }

    return Array.from(new Set(filteredGroups.map(g => g.city).filter(Boolean))).sort();
  };

  // Get all unique values (for initial state when nothing is selected)
  const allCustomerTypes = Array.from(new Set(groupTable.map(g => g.customerType).filter(Boolean))).sort();
  const allPriceTypes = Array.from(new Set(groupTable.map(g => g.priceType).filter(Boolean))).sort();
  const allConsumptionLevels = Array.from(new Set(groupTable.map(g => g.consumptionLevel).filter(Boolean))).sort();

  // Use smart filtering for dropdowns
  const customerTypes = getAvailableCustomerTypes();
  const priceTypes = getAvailablePriceTypes();
  const consumptionLevels = getAvailableConsumptionLevels();
  const availableRegions = getAvailableRegions();
  const availableSubregionsForRegion = getAvailableSubregions();
  const availableCitiesForSubregion = getAvailableCities();

  // Build location dropdowns from group table (for initial structure)
  const subregionsByRegion: Record<string, string[]> = {};
  const citiesBySubregion: Record<string, string[]> = {};

  groupTable.forEach((group) => {
    if (group.region && group.subregion) {
      if (!subregionsByRegion[group.region]) {
        subregionsByRegion[group.region] = [];
      }
      if (!subregionsByRegion[group.region].includes(group.subregion)) {
        subregionsByRegion[group.region].push(group.subregion);
      }
    }
    if (group.subregion && group.city) {
      if (!citiesBySubregion[group.subregion]) {
        citiesBySubregion[group.subregion] = [];
      }
      if (!citiesBySubregion[group.subregion].includes(group.city)) {
        citiesBySubregion[group.subregion].push(group.city);
      }
    }
  });

  // Sort subregions and cities
  Object.keys(subregionsByRegion).forEach(region => {
    subregionsByRegion[region].sort();
  });
  Object.keys(citiesBySubregion).forEach(subregion => {
    citiesBySubregion[subregion].sort();
  });

  // Use smart filtering for location dropdowns
  const availableSubregions = selectedMapRegion 
    ? availableSubregionsForRegion.filter(sub => 
        subregionsByRegion[selectedMapRegion]?.includes(sub)
      )
    : availableSubregionsForRegion;
  
  const availableCities = selectedMapSubregion
    ? availableCitiesForSubregion.filter(city =>
        citiesBySubregion[selectedMapSubregion]?.includes(city)
      )
    : availableCitiesForSubregion;

  // Validate and reset invalid filter selections when options change
  useEffect(() => {
    // Check if current customer type is still available
    if (selectedCustomerType !== "all" && customerTypes.length > 0 && !customerTypes.includes(selectedCustomerType)) {
      setSelectedCustomerType("all");
    }
    // Check if current price type is still available
    if (selectedPriceType !== "all" && priceTypes.length > 0 && !priceTypes.includes(selectedPriceType)) {
      setSelectedPriceType("all");
    }
    // Check if current consumption level is still available
    if (selectedConsumptionLevel !== "all" && consumptionLevels.length > 0 && !consumptionLevels.includes(selectedConsumptionLevel)) {
      setSelectedConsumptionLevel("all");
    }
    // Check if current region is still available
    if (selectedMapRegion && availableRegions.length > 0 && !availableRegions.includes(selectedMapRegion)) {
      setSelectedMapRegion("");
      setSelectedMapSubregion("");
      setSelectedMapCity("");
    }
    // Check if current subregion is still available
    if (selectedMapSubregion && availableSubregions.length > 0 && !availableSubregions.includes(selectedMapSubregion)) {
      setSelectedMapSubregion("");
      setSelectedMapCity("");
    }
    // Check if current city is still available
    if (selectedMapCity && availableCities.length > 0 && !availableCities.includes(selectedMapCity)) {
      setSelectedMapCity("");
    }
  }, [customerTypes, priceTypes, consumptionLevels, availableRegions, availableSubregions, availableCities]);

  // Handle location selection
  const handleRegionChange = (region: string) => {
    setSelectedMapRegion(region);
    setSelectedMapSubregion("");
    setSelectedMapCity("");
  };

  const handleSubregionChange = (subregion: string) => {
    setSelectedMapSubregion(subregion);
    setSelectedMapCity("");
  };

  const handleCityChange = (city: string) => {
    setSelectedMapCity(city);
    
    // Find a group that matches the city and optionally the selected filters
    let matchingGroup = groupTable.find(g => 
      g.city === city &&
      (!selectedCustomerType || selectedCustomerType === "all" || g.customerType === selectedCustomerType) &&
      (!selectedPriceType || selectedPriceType === "all" || g.priceType === selectedPriceType) &&
      (!selectedConsumptionLevel || selectedConsumptionLevel === "all" || g.consumptionLevel === selectedConsumptionLevel)
    );
    
    // If no matching group with filters, just find any group with that city
    if (!matchingGroup) {
      matchingGroup = groupTable.find(g => g.city === city);
    }
    
    if (matchingGroup) {
      // Only pass filters if at least one is active
      const hasActiveFilters = selectedCustomerType !== "all" || selectedPriceType !== "all" || selectedConsumptionLevel !== "all";
      const filters = hasActiveFilters ? {
        customerType: selectedCustomerType !== "all" ? selectedCustomerType : "",
        priceType: selectedPriceType !== "all" ? selectedPriceType : "",
        consumptionLevel: selectedConsumptionLevel !== "all" ? selectedConsumptionLevel : "",
      } : undefined;

      onMarkerClick(
        matchingGroup.region || selectedMapRegion,
        matchingGroup.subregion || selectedMapSubregion,
        city,
        matchingGroup.group_id,
        filters
      );
    }
  };

  // Reset all filters
  const handleResetFilters = () => {
    setSelectedCustomerType("all");
    setSelectedPriceType("all");
    setSelectedConsumptionLevel("all");
    setSelectedMapRegion("");
    setSelectedMapSubregion("");
    setSelectedMapCity("");
    
    // Clear filters from selection by calling onMarkerClick with undefined filters
    // This will hide the graph section
    if (selectedMapCity) {
      const matchingGroup = groupTable.find(g => g.city === selectedMapCity);
      if (matchingGroup) {
        onMarkerClick(
          matchingGroup.region || selectedMapRegion,
          matchingGroup.subregion || selectedMapSubregion,
          selectedMapCity,
          matchingGroup.group_id,
          undefined // No filters
        );
      }
    }
  };

  // Check if any filter is active
  const hasActiveFilters = 
    selectedCustomerType !== "all" || 
    selectedPriceType !== "all" || 
    selectedConsumptionLevel !== "all" ||
    selectedMapRegion !== "" ||
    selectedMapSubregion !== "" ||
    selectedMapCity !== "";

  // Calculate how many markers match current filters
  const allMarkers = groupTable.length > 0 ? generateMarkersFromGroups() : [];
  const filteredMarkers = allMarkers.length > 0 ? getFilteredMarkers(allMarkers) : [];
  const hasNoMarkers = filteredMarkers.length === 0 && (
    selectedMapRegion || selectedMapSubregion || selectedMapCity || 
    selectedCustomerType !== "all" || selectedPriceType !== "all" || selectedConsumptionLevel !== "all"
  );

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 border border-slate-200">
      {/* Location Selection Dropdowns */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">Select Location</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-slate-700 mb-2 text-sm font-medium">Region</label>
            <Select
              value={selectedMapRegion}
              onValueChange={handleRegionChange}
            >
              <SelectTrigger className="bg-slate-50 border-slate-300 w-full">
                <SelectValue placeholder="Choose region..." />
              </SelectTrigger>
              <SelectContent>
                {availableRegions && availableRegions.length > 0 ? (
                  availableRegions.map((region) => (
                    <SelectItem key={region} value={region}>
                      {region}
                    </SelectItem>
                  ))
                ) : (
                  <div className="px-2 py-1.5 text-sm text-slate-500">No regions available</div>
                )}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-slate-700 mb-2 text-sm font-medium">Subregion (Province)</label>
            <Select
              value={selectedMapSubregion}
              onValueChange={handleSubregionChange}
              disabled={!selectedMapRegion || availableSubregions.length === 0}
            >
              <SelectTrigger className="bg-slate-50 border-slate-300 w-full">
                <SelectValue placeholder="Choose subregion..." />
              </SelectTrigger>
              <SelectContent>
                {availableSubregions.length > 0 ? (
                  availableSubregions.map((subregion) => (
                    <SelectItem key={subregion} value={subregion}>
                      {subregion}
                    </SelectItem>
                  ))
                ) : (
                  <div className="px-2 py-1.5 text-sm text-slate-500">No subregions available</div>
                )}
              </SelectContent>
            </Select>
            </div>

          <div>
            <label className="block text-slate-700 mb-2 text-sm font-medium">City</label>
            <Select
              value={selectedMapCity}
              onValueChange={handleCityChange}
              disabled={!selectedMapSubregion || availableCities.length === 0}
            >
              <SelectTrigger className="bg-slate-50 border-slate-300 w-full">
                <SelectValue placeholder="Choose city..." />
              </SelectTrigger>
              <SelectContent>
                {availableCities.length > 0 ? (
                  availableCities.map((city) => (
                    <SelectItem key={city} value={city}>
                      {city}
                    </SelectItem>
                  ))
                ) : (
                  <div className="px-2 py-1.5 text-sm text-slate-500">No cities available</div>
                )}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Filter Dropdowns */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">Filter Markers</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-slate-700 mb-2 text-sm font-medium">Customer Type</label>
          <Select
            value={selectedCustomerType}
            onValueChange={(value) => {
              setSelectedCustomerType(value);
              // If a city is already selected, update the selection with new filters
              if (selectedMapCity && groupTable.length > 0) {
                const matchingGroup = groupTable.find(g => 
                  g.city === selectedMapCity &&
                  (value === "all" || g.customerType === value) &&
                  (selectedPriceType === "all" || g.priceType === selectedPriceType) &&
                  (selectedConsumptionLevel === "all" || g.consumptionLevel === selectedConsumptionLevel)
                );
                if (matchingGroup) {
                  // Only pass filters if at least one is active
                  const hasActiveFilters = value !== "all" || selectedPriceType !== "all" || selectedConsumptionLevel !== "all";
                  const filters = hasActiveFilters ? {
                    customerType: value !== "all" ? value : "",
                    priceType: selectedPriceType !== "all" ? selectedPriceType : "",
                    consumptionLevel: selectedConsumptionLevel !== "all" ? selectedConsumptionLevel : "",
                  } : undefined;

                  onMarkerClick(
                    matchingGroup.region || selectedMapRegion,
                    matchingGroup.subregion || selectedMapSubregion,
                    selectedMapCity,
                    matchingGroup.group_id,
                    filters
                  );
                }
              }
            }}
          >
            <SelectTrigger className="bg-slate-50 border-slate-300 w-full">
              <SelectValue placeholder="All" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              {customerTypes.length > 0 ? (
                customerTypes.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))
              ) : (
                <div className="px-2 py-1.5 text-sm text-slate-500">No customer types available</div>
              )}
            </SelectContent>
          </Select>
      </div>

        <div>
          <label className="block text-slate-700 mb-2 text-sm font-medium">Price Type</label>
          <Select
            value={selectedPriceType}
            onValueChange={(value) => {
              setSelectedPriceType(value);
              // If a city is already selected, update the selection with new filters
              if (selectedMapCity && groupTable.length > 0) {
                const matchingGroup = groupTable.find(g => 
                  g.city === selectedMapCity &&
                  (selectedCustomerType === "all" || g.customerType === selectedCustomerType) &&
                  (value === "all" || g.priceType === value) &&
                  (selectedConsumptionLevel === "all" || g.consumptionLevel === selectedConsumptionLevel)
                );
                if (matchingGroup) {
                  // Only pass filters if at least one is active
                  const hasActiveFilters = selectedCustomerType !== "all" || value !== "all" || selectedConsumptionLevel !== "all";
                  const filters = hasActiveFilters ? {
                    customerType: selectedCustomerType !== "all" ? selectedCustomerType : "",
                    priceType: value !== "all" ? value : "",
                    consumptionLevel: selectedConsumptionLevel !== "all" ? selectedConsumptionLevel : "",
                  } : undefined;

                  onMarkerClick(
                    matchingGroup.region || selectedMapRegion,
                    matchingGroup.subregion || selectedMapSubregion,
                    selectedMapCity,
                    matchingGroup.group_id,
                    filters
                  );
                }
              }
            }}
          >
            <SelectTrigger className="bg-slate-50 border-slate-300 w-full">
              <SelectValue placeholder="All" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              {priceTypes.length > 0 ? (
                priceTypes.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))
              ) : (
                <div className="px-2 py-1.5 text-sm text-slate-500">No price types available</div>
              )}
            </SelectContent>
          </Select>
        </div>
        
        <div>
          <label className="block text-slate-700 mb-2 text-sm font-medium">Consumption Level</label>
          <Select
            value={selectedConsumptionLevel}
            onValueChange={(value) => {
              setSelectedConsumptionLevel(value);
              // If a city is already selected, update the selection with new filters
              if (selectedMapCity && groupTable.length > 0) {
                const matchingGroup = groupTable.find(g => 
                  g.city === selectedMapCity &&
                  (selectedCustomerType === "all" || g.customerType === selectedCustomerType) &&
                  (selectedPriceType === "all" || g.priceType === selectedPriceType) &&
                  (value === "all" || g.consumptionLevel === value)
                );
                if (matchingGroup) {
                  // Only pass filters if at least one is active
                  const hasActiveFilters = selectedCustomerType !== "all" || selectedPriceType !== "all" || value !== "all";
                  const filters = hasActiveFilters ? {
                    customerType: selectedCustomerType !== "all" ? selectedCustomerType : "",
                    priceType: selectedPriceType !== "all" ? selectedPriceType : "",
                    consumptionLevel: value !== "all" ? value : "",
                  } : undefined;

                  onMarkerClick(
                    matchingGroup.region || selectedMapRegion,
                    matchingGroup.subregion || selectedMapSubregion,
                    selectedMapCity,
                    matchingGroup.group_id,
                    filters
                  );
                }
              }
            }}
          >
            <SelectTrigger className="bg-slate-50 border-slate-300 w-full">
              <SelectValue placeholder="All" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              {consumptionLevels.length > 0 ? (
                consumptionLevels.map((level) => (
                  <SelectItem key={level} value={level}>
                    {level}
                  </SelectItem>
                ))
              ) : (
                <div className="px-2 py-1.5 text-sm text-slate-500">No consumption levels available</div>
              )}
            </SelectContent>
          </Select>
          </div>
              </div>
            </div>

      {/* Reset Filters Button */}
      <div className="mb-6 flex justify-end">
        <button
          onClick={handleResetFilters}
          disabled={!hasActiveFilters}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            hasActiveFilters
              ? "bg-slate-200 hover:bg-slate-300 text-slate-700 cursor-pointer"
              : "bg-slate-100 text-slate-400 cursor-not-allowed"
          }`}
        >
          Reset Filters
        </button>
          </div>

      {/* Warning message when no markers match */}
      {hasNoMarkers && (
        <div className="mb-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <div className="flex items-start">
            <svg className="w-5 h-5 text-amber-600 mt-0.5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <h4 className="text-sm font-semibold text-amber-800 mb-1">No markers match your filters</h4>
              <p className="text-sm text-amber-700">
                The selected combination of location and filters does not match any markers. Please adjust your filters or reset them to see all markers.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Marker count info */}
      {!hasNoMarkers && filteredMarkers.length > 0 && (
        <div className="mb-4 text-sm text-slate-600">
          Showing <span className="font-semibold text-slate-900">{filteredMarkers.length}</span> of <span className="font-semibold text-slate-900">{allMarkers.length}</span> markers
      </div>
      )}

      <div
        className="w-full border border-slate-300 rounded-lg overflow-hidden relative"
        style={{ height: "500px" }}
      >
        {/* Top popup container */}
        <div
          ref={topPopupRef}
          className="top-popup-container"
          style={{
            position: 'absolute',
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1000,
            display: 'none',
            maxWidth: '90%',
            width: 'auto'
          }}
        />
        {/* View mode toggle button */}
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            zIndex: 1000,
            display: 'flex',
            gap: '4px',
            background: '#0f172a',
            padding: '4px',
            borderRadius: '8px',
            border: '2px solid #334155',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
          }}
        >
          <button
            onClick={() => {
              setViewMode('hourly');
              // Close popup when switching view mode
              if (topPopupRef.current) {
                topPopupRef.current.style.display = 'none';
              }
            }}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '6px',
              background: viewMode === 'hourly' ? '#3B82F6' : 'transparent',
              color: viewMode === 'hourly' ? 'white' : '#94a3b8',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              if (viewMode !== 'hourly') {
                e.currentTarget.style.background = '#1e293b';
              }
            }}
            onMouseOut={(e) => {
              if (viewMode !== 'hourly') {
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            Hourly
          </button>
          <button
            onClick={() => {
              setViewMode('monthly');
              // Close popup when switching view mode
              if (topPopupRef.current) {
                topPopupRef.current.style.display = 'none';
              }
            }}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '6px',
              background: viewMode === 'monthly' ? '#3B82F6' : 'transparent',
              color: viewMode === 'monthly' ? 'white' : '#94a3b8',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              if (viewMode !== 'monthly') {
                e.currentTarget.style.background = '#1e293b';
              }
            }}
            onMouseOut={(e) => {
              if (viewMode !== 'monthly') {
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            Monthly
          </button>
        </div>
        {/* Map container */}
        <div
          ref={ref}
          className="w-full h-full"
        />
      </div>
      <style>{`
        .custom-marker-clickable {
          pointer-events: auto !important;
          cursor: pointer !important;
        }
        .custom-marker-clickable > div {
          position: relative !important;
          width: 100% !important;
          height: 100% !important;
        }
        .custom-marker-clickable * {
          pointer-events: none !important;
        }
        .custom-marker-clickable:hover {
          opacity: 0.9;
        }
        .maplibregl-marker {
          cursor: pointer !important;
          pointer-events: auto !important;
        }
        .maplibregl-marker .custom-marker-clickable {
          pointer-events: auto !important;
        }
        .maplibregl-marker-container {
          pointer-events: auto !important;
        }
        .custom-popup {
          max-width: 300px !important;
        }
        .maplibregl-popup {
          max-width: 300px !important;
        }
        .maplibregl-popup-content {
          max-width: 100% !important;
          word-wrap: break-word;
        }
        /* Ensure popup stays within map bounds */
        .maplibregl-popup-anchor-bottom .maplibregl-popup-tip {
          border-top-color: #0f172a;
        }
        .maplibregl-popup-anchor-top .maplibregl-popup-tip {
          border-bottom-color: #0f172a;
        }
        .maplibregl-popup-anchor-left .maplibregl-popup-tip {
          border-right-color: #0f172a;
        }
        .maplibregl-popup-anchor-right .maplibregl-popup-tip {
          border-left-color: #0f172a;
        }
      `}</style>
    </div>
  );
}
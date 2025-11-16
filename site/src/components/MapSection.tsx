import React, { useEffect, useRef, useState } from "react";
import maplibregl, { Map } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { loadGroupTable, GroupTableRow } from "../utils/csvLoader";
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
    }
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

const getIntensity = (v: number) =>
  v > 1000 ? 4 : v > 700 ? 3 : v > 500 ? 2 : 1;

const getTrendIcon = (t: string) =>
  t === "up" ? "↑" : t === "down" ? "↓" : "→";

export function MapSection({
  selectedRegion,
  selectedCity,
  onMarkerClick,
}: MapSectionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<Map | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [groupTable, setGroupTable] = useState<GroupTableRow[]>([]);
  const [regionMapFromGroups, setRegionMapFromGroups] = useState<Record<string, string>>({});
  
  // Filter states
  const [selectedCustomerType, setSelectedCustomerType] = useState<string>("all");
  const [selectedPriceType, setSelectedPriceType] = useState<string>("all");
  const [selectedConsumptionLevel, setSelectedConsumptionLevel] = useState<string>("all");
  
  // Location selection states
  const [selectedMapRegion, setSelectedMapRegion] = useState<string>("");
  const [selectedMapSubregion, setSelectedMapSubregion] = useState<string>("");
  const [selectedMapCity, setSelectedMapCity] = useState<string>("");

  const popupRef = useRef<maplibregl.Popup | null>(null);

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
      const styleUrl = `https://tiles.stadiamaps.com/styles/stamen_toner.json?api_key=${apiKey}`;

      console.log("Initializing map with container:", {
        width: ref.current!.offsetWidth,
        height: ref.current!.offsetHeight
      });

      try {
        map.current = new maplibregl.Map({
          container: ref.current!,
          style: styleUrl,
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

        map.current.on("load", () => {
          console.log("Map loaded successfully");
          setLoaded(true);
          if (map.current) {
            map.current.resize();
            console.log("Map resized after load");
          }
        });

        map.current.on("error", (e: any) => {
          console.error("Map error:", e);
        });

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

      // Estimate consumption based on consumption level (for visualization)
      const consumptionMap: Record<string, number> = {
        "Low": 500,
        "Medium": 750,
        "High": 1200,
      };
      const estimatedConsumption = consumptionMap[group.consumptionLevel] || 700;

      return {
        id: String(group.group_id),
        name: group.city || group.subregion,
        subregion: group.subregion,
        city: group.city,
        groupId: group.group_id,
        lng: coords.lng,
        lat: coords.lat,
        predictedConsumption: estimatedConsumption,
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

  // Create markers when map is loaded AND group table is loaded
  useEffect(() => {
    if (!loaded || !map.current) return;
    
    // Wait for group table to load
    if (groupTable.length === 0) {
      return;
    }

    // Clear existing markers first
    const existingMarkers = document.querySelectorAll('.maplibregl-marker');
    existingMarkers.forEach(m => m.remove());

    // Generate all markers from group table
    const allMarkers = generateMarkersFromGroups();
    const filteredMarkers = getFilteredMarkers(allMarkers);

    console.log(`Displaying ${filteredMarkers.length} of ${allMarkers.length} markers`);

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

        const intensity = getIntensity(m.predictedConsumption);

        // Create inner content wrapper with relative positioning
        const contentWrapper = document.createElement("div");
        contentWrapper.style.position = "relative";
        contentWrapper.style.width = "100%";
        contentWrapper.style.height = "100%";
        contentWrapper.style.pointerEvents = "none";
        
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
              border:2px solid ${isRegionSelected ? "#6A38FF" : "#1e293b"};
              opacity:${opacity};
              pointer-events:none;
            "></div>
          `;
            })
            .join("") +
          `
          <div style="
            position:absolute;
            top:50%; left:50%;
            transform:translate(-50%,-50%);
            width:34px; height:34px;
            border-radius:50%;
            border:3px solid ${isRegionSelected ? "#6A38FF" : "#1e293b"};
            background:white;
            box-shadow:0 2px 6px rgba(0,0,0,0.2);
            pointer-events:none;
          ">
            <div style="
              position:absolute;
              top:50%; left:50%;
              transform:translate(-50%,-50%);
              width:12px; height:12px;
              border-radius:50%;
              background:${isRegionSelected ? "#6A38FF" : "#1e293b"};
            "></div>
          </div>
          <div style="
            position:absolute;
            top:70px; left:50%;
            transform:translateX(-50%);
            background:white;
            font-size:12px;
            padding:4px 6px;
            border-radius:4px;
            border:1px solid #ddd;
            color:#1e293b;
            white-space:nowrap;
            pointer-events:none;
          ">
            ${m.predictedConsumption} kWh
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
          
          // Only pass filters if at least one is active
          const hasActiveFilters = selectedCustomerType !== "all" || selectedPriceType !== "all" || selectedConsumptionLevel !== "all";
          const filters = hasActiveFilters ? {
            customerType: selectedCustomerType !== "all" ? selectedCustomerType : "",
            priceType: selectedPriceType !== "all" ? selectedPriceType : "",
            consumptionLevel: selectedConsumptionLevel !== "all" ? selectedConsumptionLevel : "",
          } : undefined;

          onMarkerClick(
            groupToUse?.region || finalRegion, 
            groupToUse?.subregion || finalSubregion, 
            finalCity, 
            groupIdToUse,
            filters
          );

          // Remove existing popup
          popupRef.current?.remove();

          // Create new popup with group table data
          popupRef.current = new maplibregl.Popup({
            offset: 20,
            closeButton: true,
          })
            .setLngLat([m.lng, m.lat])
            .setHTML(`
              <div style="padding:8px 6px;min-width:200px;font-size:14px;color:white;background:#0f172a;border-radius:6px;border:1px solid #334155;">
                <b style="font-size:16px;">${finalCity || m.name}</b><br/>
                <span style="color:#cbd5e1;font-size:12px;margin-top:4px;display:block;">
                  <strong>Region:</strong> ${finalRegion}<br/>
                  <strong>Subregion:</strong> ${finalSubregion}<br/>
                  <strong>Group ID:</strong> ${m.groupId}
                </span>
                ${groupInfo ? `
                <div style="color:#94a3b8;font-size:11px;margin-top:6px;padding-top:6px;border-top:1px solid #334155;">
                  <strong>Customer:</strong> ${groupInfo.customerType}<br/>
                  <strong>Price Type:</strong> ${groupInfo.priceType}<br/>
                  <strong>Consumption:</strong> ${groupInfo.consumptionLevel}
                </div>
                ` : ''}
                <hr style="border-color:#334155;margin:8px 0;" />
                <span style="color:#cbd5e1;font-size:12px;">Daily prediction</span>
                <div style="margin-top:4px;">
                  <b style="font-size:15px;">${m.predictedConsumption} kWh/day</b>
                  <span style="margin-left:6px;color:#94a3b8;">${getTrendIcon(m.trend)}</span>
                </div>
              </div>
            `)
            .addTo(map.current!);
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
        const marker = new maplibregl.Marker({
          element: el,
          anchor: 'center'
        })
          .setLngLat([m.lng, m.lat])
          .addTo(map.current!);
        
        // Store marker reference for potential cleanup
        (el as any)._marker = marker;
      });
  }, [loaded, groupTable, regionMapFromGroups, selectedRegion, selectedCustomerType, selectedPriceType, selectedConsumptionLevel, selectedMapRegion, selectedMapSubregion, selectedMapCity, onMarkerClick]);

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
        ref={ref}
        className="w-full border border-slate-300 rounded-lg overflow-hidden"
        style={{ height: "500px" }}
      />
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
      `}</style>
    </div>
  );
}
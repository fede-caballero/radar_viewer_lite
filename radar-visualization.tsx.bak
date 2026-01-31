'use client'

import { useState, useEffect, useMemo, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, RotateCcw, Calendar, Clock } from "lucide-react"
import { ImageWithBounds } from "@/lib/api"
import Map, { Source, Layer, NavigationControl, ScaleControl, FullscreenControl, GeolocateControl, MapRef } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'

interface RadarVisualizationProps {
  inputFiles: ImageWithBounds[]
  predictionFiles: ImageWithBounds[]
  isProcessing: boolean
}

const INITIAL_VIEW_STATE = {
  longitude: -68.016,
  latitude: -34.647,
  zoom: 8
};

// Dark Matter style for a premium look
const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

function haversineDistance(coords1: { lat: number, lon: number }, coords2: { lat: number, lon: number }) {
  const toRad = (x: number) => x * Math.PI / 180;
  const R = 6371; // km
  const dLat = toRad(coords2.lat - coords1.lat);
  const dLon = toRad(coords2.lon - coords1.lon);
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(coords1.lat)) * Math.cos(toRad(coords2.lat)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export function RadarVisualization({ inputFiles, predictionFiles, isProcessing }: RadarVisualizationProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [boundariesData, setBoundariesData] = useState<any>(null)
  const [userLocation, setUserLocation] = useState<{ latitude: number, longitude: number } | null>(null)
  const mapRef = useRef<MapRef>(null)

  // Merge frames: Inputs + Predictions
  const frames = useMemo(() => {
    return [...inputFiles, ...predictionFiles];
  }, [inputFiles, predictionFiles]);

  const totalFrames = frames.length;

  // Load boundaries
  useEffect(() => {
    fetch('/boundaries.json')
      .then(res => res.json())
      .then(data => setBoundariesData(data))
      .catch(err => console.error("Failed to load boundaries", err));
  }, []);

  // Animation logic
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying && totalFrames > 0) {
      interval = setInterval(() => {
        setCurrentFrameIndex((prev) => {
          if (prev >= totalFrames - 1) return 0; // Loop back to start
          return prev + 1;
        })
      }, 2000) // 2000ms per frame (slower for better loading on high latency)
    }
    return () => clearInterval(interval)
  }, [isPlaying, totalFrames])

  // Reset when data changes significantly
  useEffect(() => {
    if (totalFrames > 0 && currentFrameIndex >= totalFrames) {
      setCurrentFrameIndex(0);
    }
  }, [totalFrames]);

  const togglePlay = () => setIsPlaying(!isPlaying)
  const resetAnimation = () => { setIsPlaying(false); setCurrentFrameIndex(0); }

  const currentImage = frames[currentFrameIndex];
  const isPrediction = currentFrameIndex >= inputFiles.length;

  const getImageCoordinates = (image: ImageWithBounds | undefined) => {
    if (!image?.bounds) return undefined;
    const b = image.bounds as any;
    const p1 = b[0];
    const p2 = b[1];

    const minLat = Math.min(p1[0], p2[0]);
    const maxLat = Math.max(p1[0], p2[0]);
    const minLon = Math.min(p1[1], p2[1]);
    const maxLon = Math.max(p1[1], p2[1]);

    return [
      [minLon, maxLat], // TL
      [maxLon, maxLat], // TR
      [maxLon, minLat], // BR
      [minLon, minLat]  // BL
    ] as [[number, number], [number, number], [number, number], [number, number]];
  }

  const imageCoordinates = useMemo(() => getImageCoordinates(currentImage), [currentImage]);

  // Calculate center of current image for distance
  const stormCenter = useMemo(() => {
    if (!currentImage?.bounds) return null;
    const b = currentImage.bounds as any;
    // Assuming bounds are [[lat1, lon1], [lat2, lon2]] or similar based on usage
    // Code above uses p1[0] as lat, p1[1] as lon
    const p1 = b[0];
    const p2 = b[1];
    return {
      lat: (p1[0] + p2[0]) / 2,
      lon: (p1[1] + p2[1]) / 2
    };
  }, [currentImage]);

  const distanceToStorm = useMemo(() => {
    if (!userLocation || !stormCenter) return null;
    return haversineDistance(
      { lat: userLocation.latitude, lon: userLocation.longitude },
      stormCenter
    ).toFixed(1);
  }, [userLocation, stormCenter]);

  const boundaryLayerStyle = {
    id: 'boundaries-layer',
    type: 'line',
    paint: {
      'line-color': '#facc15',
      'line-width': 2,
      'line-opacity': 0.6
    }
  } as const;

  // Calculate time label (approximate based on index)
  // Assuming inputs are every 15 min and predictions every 3 min (based on previous context)
  // But for simplicity in UI, we'll just show "Past" vs "Forecast +X min"
  const getTimeLabel = () => {
    if (currentImage?.target_time) {
      return `Pronóstico ${currentImage.target_time}`;
    }
    if (!isPrediction) {
      return `Radar Pasado (${currentFrameIndex + 1}/${inputFiles.length})`;
    } else {
      const predIndex = currentFrameIndex - inputFiles.length;
      return `Pronóstico T+${(predIndex + 1) * 3} min`;
    }
  };

  return (
    <div className="relative w-full h-full bg-black">
      <Map
        ref={mapRef}
        initialViewState={INITIAL_VIEW_STATE}
        style={{ width: '100%', height: '100%' }}
        mapStyle={MAP_STYLE}
        attributionControl={false}
      >
        <NavigationControl position="top-right" />
        <GeolocateControl
          position="top-right"
          trackUserLocation={true}
          showUserLocation={true}
          onGeolocate={(evt) => {
            setUserLocation({
              latitude: evt.coords.latitude,
              longitude: evt.coords.longitude
            });
          }}
        />
        <ScaleControl />
        <FullscreenControl position="top-right" />

        {boundariesData && (
          <Source id="boundaries-source" type="geojson" data={boundariesData}>
            <Layer {...boundaryLayerStyle} />
          </Source>
        )}

        {/* Logo Overlay */}
        <div className="absolute top-4 left-4 z-10 pointer-events-none flex flex-col items-center">
          <img src="/logo.png" alt="Hailcast Logo" className="w-24 h-24 object-contain drop-shadow-lg opacity-90" />
          <p className="text-[10px] text-white/90 font-medium mt-1 drop-shadow-md bg-black/40 px-2 py-0.5 rounded-full backdrop-blur-sm tracking-wide">
            Sistema de Predicción Meteorológica
          </p>
          {distanceToStorm && (
            <div className="mt-2 bg-red-500/80 text-white px-3 py-1 rounded-md text-xs font-bold backdrop-blur-md shadow-lg animate-pulse">
              Distancia al Centro de Tormenta: {distanceToStorm} km
            </div>
          )}
        </div>



        {currentImage && imageCoordinates && (
          <Source
            id="radar-source"
            type="image"
            url={currentImage.url}
            coordinates={imageCoordinates}
          >
            <Layer
              id="radar-layer"
              type="raster"
              paint={{
                "raster-opacity": 0.8,
                "raster-fade-duration": 0
              }}
            />
          </Source>
        )}

        {/* Unified Timeline Control Bar */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 via-black/60 to-transparent pb-8 pt-12">
          <div className="max-w-4xl mx-auto w-full flex flex-col gap-2">

            {/* Time Label & Status */}
            <div className="flex justify-between items-end px-2 mb-1">
              <div className="flex flex-col">
                <span className={`text-xs font-bold uppercase tracking-wider ${isPrediction ? 'text-primary' : 'text-muted-foreground'}`}>
                  {isPrediction ? 'Modelo Predictivo' : 'Datos Observados'}
                </span>
                <span className="text-2xl font-light text-foreground flex items-center gap-2">
                  {isPrediction ? <Clock className="w-5 h-5 text-primary" /> : <Calendar className="w-5 h-5 text-muted-foreground" />}
                  {getTimeLabel()}
                </span>
              </div>

              <div className="flex gap-2">
                <Button
                  size="icon"
                  variant="secondary"
                  className="h-10 w-10 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 shadow-[0_0_15px_rgba(133,153,51,0.5)]"
                  onClick={togglePlay}
                >
                  {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5 ml-1" />}
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-10 w-10 rounded-full text-muted-foreground hover:text-foreground hover:bg-white/10"
                  onClick={resetAnimation}
                >
                  <RotateCcw className="h-5 w-5" />
                </Button>
              </div>
            </div>

            {/* Slider Track */}
            <div className="relative h-6 flex items-center group">
              {/* Background Track with "Past" vs "Future" distinction */}
              <div className="absolute inset-x-0 h-1.5 bg-white/10 rounded-full overflow-hidden flex">
                <div
                  className="h-full bg-white/20"
                  style={{ width: `${(inputFiles.length / Math.max(1, totalFrames)) * 100}%` }}
                />
                <div
                  className="h-full bg-primary/20"
                  style={{ width: `${(predictionFiles.length / Math.max(1, totalFrames)) * 100}%` }}
                />
              </div>

              <Slider
                value={[currentFrameIndex]}
                onValueChange={(value) => { setIsPlaying(false); setCurrentFrameIndex(value[0]); }}
                max={Math.max(0, totalFrames - 1)}
                step={1}
                className="cursor-pointer z-10"
              />
            </div>

            {/* Ticks/Labels under slider */}
            <div className="flex justify-between text-[10px] text-muted-foreground px-1 font-mono uppercase tracking-widest">
              <span>Pasado</span>
              <span>Ahora</span>
              <span>Futuro</span>
            </div>

          </div>
        </div>
      </Map>
    </div>
  )
}

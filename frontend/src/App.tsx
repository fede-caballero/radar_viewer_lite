import { useEffect, useState } from "react"
import { RadarVisualization } from "@/components/radar-visualization"
import type { ImageWithBounds } from "@/lib/api"

function App() {
  const [inputFiles, setInputFiles] = useState<ImageWithBounds[]>([])
  const [predictionFiles] = useState<ImageWithBounds[]>([])

  const fetchData = () => {
    fetch('data/data.json') // Fetch from relative path
      .then(res => res.json())
      .then((data: ImageWithBounds[]) => {
        const fixedData = data.map(item => ({
          ...item,
          url: `data/${item.url}`
        }))
        // Only update if data changed (simple length check or deep comparison could be better but this is Lite)
        setInputFiles(prev => {
          if (JSON.stringify(prev) !== JSON.stringify(fixedData)) {
            return fixedData;
          }
          return prev;
        })
      })
      .catch(err => console.error("Failed to load radar data", err))
  }

  useEffect(() => {
    fetchData(); // Initial load
    const interval = setInterval(fetchData, 60000); // Poll every 60s
    return () => clearInterval(interval);
  }, [])

  return (
  return (
    <div className="fixed inset-0 w-screen h-screen m-0 p-0 overflow-hidden" style={{ height: '100vh' }}>
      <RadarVisualization
        inputFiles={inputFiles}
        predictionFiles={predictionFiles}
        isProcessing={false}
      />
    </div>
  )
}

export default App

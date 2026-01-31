import { useEffect, useState } from "react"
import { RadarVisualization } from "@/components/radar-visualization"
import type { ImageWithBounds } from "@/lib/api"

function App() {
  const [inputFiles, setInputFiles] = useState<ImageWithBounds[]>([])
  const [predictionFiles] = useState<ImageWithBounds[]>([])

  useEffect(() => {
    fetch('data/data.json') // Fetch from relative path (important for GitHub Pages subdir)
      .then(res => res.json())
      .then((data: ImageWithBounds[]) => {
        // Filter or sort if needed. Assuming backend provides sorted list.
        // Identify predictions vs inputs if distinguishable? 
        // Current backend puts everything in one list. logic might need adjustment if predictions are separate.
        // For now, assume all are inputs or mix.
        // user prompt said: "Backend... hace commit... data.json".
        setInputFiles(data)
      })
      .catch(err => console.error("Failed to load radar data", err))
  }, [])

  return (
    <div className="w-screen h-screen">
      <RadarVisualization
        inputFiles={inputFiles}
        predictionFiles={predictionFiles}
        isProcessing={false}
      />
    </div>
  )
}

export default App

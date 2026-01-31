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
        // Fix URLs to be relative to the app root (data/images/...)
        // Backend provides "images/file.png", but strictly it lives in "data/images/file.png"
        const fixedData = data.map(item => ({
          ...item,
          url: `data/${item.url}`
        }))
        setInputFiles(fixedData)
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

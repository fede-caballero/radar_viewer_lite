export interface ImageWithBounds {
    url: string;
    bounds: [[number, number], [number, number]]; // [lat, lon], [lat, lon] ?? No, maplibre uses [lon, lat]. Checking component.
    // Component usage:
    // p1 = b[0], p2 = b[1]
    // bounds are [[lat, lon], [lat, lon]] usually in LROSE logic from python script:
    // bounds = [[sw_corner[1], sw_corner[0]], [ne_corner[1], ne_corner[0]]] -> [[lat, lon], [lat, lon]]
    target_time?: string;
}

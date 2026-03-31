# BIMMY

BIMMY (Building Information Modeling Management Yard) is a lightweight OpenGL BIM MVP focused on the relationship between 3D geometry and live cost tracking.

## MVP Features

- Parametric primitives: Cuboid, Cylinder, Prism
- Scene component model with per-object metadata and IDs
- OpenGL renderer with orbit camera, depth testing, and material shading
- Mouse picking using Ray-AABB intersection
- Interactive transforms: translate, rotate, scale
- Real-time Bill of Materials (BOM) and grand total cost updates
- Dual-pane UI (3D viewport + Inspector/Inventory dashboard)

## Build

### Requirements

- CMake 3.20+
- C++20 compiler (GCC, Clang, or MSVC)
- Internet access on first configure (dependencies fetched automatically)

### Linux/macOS

```bash
cmake -S . -B build
cmake --build build -j
./build/bimmy
```

### Windows (PowerShell)

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\bimmy.exe
```

## Controls

- Right mouse drag: orbit camera
- F: toggle free-look mode (mouse look without holding RMB)
- Mouse wheel: zoom
- W/A/S/D: move camera target on ground plane
- Q/E: move camera target down/up
- Shift: faster keyboard movement
- Hold Alt + Arrow keys/PageUp/PageDown: translate selected component
- Hold Alt + I/K/J/L/U/O: rotate selected component (X/Y/Z)
- Hold Alt + [ or ]: uniformly scale selected component
- Left mouse click: select component (raycast)
- Inspector panel: edit dimensions, transform, material
- Inventory dashboard: view per-material volume/cost and grand total

## Shader Assets

- Runtime shaders are stored in `shaders/bimmy.vert` and `shaders/bimmy.frag`.
- CMake copies these files next to the executable under `shaders/` at build time.
- Renderer falls back to built-in shader source if files are missing, so startup remains robust.

## Notes on BOM Logic

- Cuboid volume: `x * y * z`
- Cylinder volume: `pi * r^2 * h`
- Triangular prism volume: `0.5 * width * height * depth`
- Cost: `volume * material_cost_per_unit_volume`

All BOM values update every frame based on current component transforms and dimensions.

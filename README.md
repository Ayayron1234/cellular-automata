## Falling sand

### TODO: 

| Deadline   | Conservative Time Estimate | Task                                                                                             |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------ |
| 2024.04.10 | 12h                        | Fixing remaining multithreading bugs (2-6h), cell velocity(2-4h), different particle sizes(2-3h) |
| 2024.04.17 | 4h                         | Dirty rectangle                                                                                  |
| 2024.04.24 | 6h                         | Better UI, way to draw or create cells with texture                                              |
| 2024.05.01 | 6h                         | More cell types (eg.: dirt, stone, vegetation, fire, lava)                                       |
| 2024.05.08 | 4h                         | More cell types (eg.: dirt, stone, vegetation, fire, lava)                                       |
| 2024.05.15 | ?                          | Bug fixes, code cleanup                                                                          |
| 2024.05.22 | ?                          | Bug fixes, presentation                                                                          |

### Napló

| Dátum      | idő   | feladat                        |
| ---------- | ----- | ------------------------------ |
| 2024.03.23 | 3 óra | fixed bugs with chunk updating |
| 2024.03.30 | 2 óra | Multithreading hibák javítása. Valamivel stabilabb a rendszer. 
| 2024.04.02 | 3 óra | Thread pool rework. Ez az első thread pool amit csináltam, szóval csak idő közben értettem meg a c++ multithreadinghez tartozó osztályok működését. Sikerült megelőzni a random lefagyásokat, de rájöttem hogy vannak hibák amik  a jelenlegi végrehajtással nem javítható. (a chunkok feldolgozása 3 szinkronizált fázisban kell történjen. )
| 2024.04.04 | 2 óra | Új chunk feldolgozási rendszer tervezése. (vázlatos szekvencia diagram)
| 2024.04.05 | 5 óra | Thread pool rework. Implementáltam a korábbi tervet, így a multi threading már jól működik, viszont nem vagyok elégedett a jelenlegi rendereléssel ami CUDA-t használ, mert feleslegesen sokszor passzolok adatot videó memóriába és vissza. 
| 2024.04.08 | 3 óra | Rendering rework to only use SDL_Renderer. A chunkokat SDL_Texture textúrába rajzoltam cpu-ról multithreading-el. Hiba: SDL nem tudja kezelni a több szálról érkező hívásokat. 
| 2024.04.18 | 12 óra | Rendering reworked to opengl and reorganized Chunk class to have less responsibility (Created: ChunkView, ChunkState, WorldView). A chunkok celláinak renderelését megvalósítottam opengl-ben. (Chunk határokat még nem tud rajzolni, és minden chunk egy külön draw call amit később lehet még javítani. )
| 2024.04.19 | 3 óra | Chunk border drawing implemented with opengl, minor bug fixes. (ideiglenes megoldás, később majd chunkoktól független, optimalizált négyzetrajzolással kéne megoldani. )
| 2024.04.20 | 5 óra | Hot shader reloading. Shaderek autómatikusan frissülnek, ha az őket tartalmazó file-ba írok. Letisztítottam kicsit a Shader osztály implementációját. 
| 2024.04.23 | 6 óra | Bug hunting. Javítottam egy hibát a szomszédos chunkok frissítésével. Javítás egyszerű volt, de nehezen találtam meg a hiba okát. 
| 2024.04.24 | 3 óra | Fixed occasional freezing on chunk creation. Fixed some memory leaks. Started removing CUDA integration which was needed for earlyer rendering implementation. 
| 2024.05.06 | 8 óra | glsl cell shade error fix, Eraser, SolidBrush, RandomizerBrush
| 2024.05.07 | 8 óra | Texture, TexturedBrush, imgui brush widget, Cell::update(), bitmap hot reload (for color palette and textures), isFreefalling
| 2024.05.08 | 8 óra | textured brush imgui widget, PerformanceMonitor, GUI, main file cleanup

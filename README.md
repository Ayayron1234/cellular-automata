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

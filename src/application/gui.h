#pragma once
#include "../utils.h"

class GUI {
public:
	GUI() {
		initBrushes();
	}

    void createWindow() {
        if (!IO::OpenWindow(m_options.windowWidth, m_options.windowHeight))
            exit(1);
    }

    std::string getWorldPath() {
        static std::string c_worldPath{};

        // Return world path if was allready provided
        if (!c_worldPath.empty())
            return c_worldPath;

        // Set default world in debug mode
        #ifdef _DEBUG
        c_worldPath = "data/worlds/dbgTest";
        return c_worldPath;
        #endif // _DEBUG

        // Read world name from the console
        std::cout << "World name: ";
        std::string worldName; std::getline(std::cin, worldName);

        // If name was empty set it to test
        if (worldName.empty())
            worldName = "test";

        c_worldPath = "data/worlds/" + worldName;
        return c_worldPath;
    }

	void handleInputs(World& world) {
        // Handle events
        IO::HandleEvents();

        // Handle window resize
        if (IO::Resized()) {
            m_options.windowWidth = IO::GetWindowWidth();
            m_options.windowHeight = IO::GetWindowHeight();
        }

        vec2 normalizedMousePos = IO::NormalizePixel((int)IO::GetMousePos().x, (int)IO::GetMousePos().y);
        vec2 mouseWorldPos = m_options.camera.screenToWorld(m_options.windowWidth, m_options.windowHeight, IO::GetMousePos());
        CellCoord mouseCellCoord{ (int)floor(mouseWorldPos.x), (int)floor(mouseWorldPos.y) };

        // Insert break point to hovered chunks process method with the B key
        #ifdef _DEBUG
        if (IO::IsKeyDown(SDL_SCANCODE_B)) {
            std::cout << "yes" << std::endl;
            ChunkCoord coord = CellCoord(mouseWorldPos).getChunkCoord();
            Chunk* chunk = world.getChunkIfPopulated(coord);
            if (chunk)
                chunk->_dbgInsertBreakOnProcess();
        }
        #endif

        // Update brush type with hotkeys
        if (IO::IsKeyDown(SDL_SCANCODE_E)) m_brushSelectedIndex = 0;
        if (IO::IsKeyDown(SDL_SCANCODE_S)) m_brushSelectedIndex = 1;
        if (IO::IsKeyDown(SDL_SCANCODE_R)) m_brushSelectedIndex = 2;
        if (IO::IsKeyDown(SDL_SCANCODE_T)) m_brushSelectedIndex = 3;
        if (m_brush)
            m_brush->setType(Cell::Type(m_options.brushCellType));

        // Copy cell type to brush with middle mouse click
        if (IO::MouseClicked(SDL_BUTTON_MIDDLE)) {
            Cell target = world.getCell(mouseCellCoord);
            if (target.type != Cell::Type::AIR)
                m_options.brushCellType = (int)target.type;
        }

        // Draw with brush when right button is pressed
        if (IO::IsButtonDown(SDL_BUTTON_RIGHT) && m_brush != nullptr)
            m_brush->draw(mouseCellCoord, world);

        moveCamera(mouseWorldPos, normalizedMousePos);
	}

    void render() {
        showImGuiWindow();

        IO::Render();

        if (GLenum error = glGetError(); error != GL_NO_ERROR)
            std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
    }

    Options& options() {
        return m_options;
    }

    void loadOptions(const std::string& path) {
        m_options = Options(JsonFileResource("data/options.json").value());
    }

    ~GUI() {
        IO::Quit();
    }

private:
	using BrushVec = std::vector<Brush*>;

	Options		m_options;
	bool		m_cameraMovementDisabled;

	Brush*		m_brush = nullptr;
	BrushVec	m_brushes;
	int			m_brushSelectedIndex = 2;

	void initBrushes() {
		Cell::Type selectedCellType = *(Cell::Type*)(&m_options.brushCellType);
		m_brushes = BrushVec{
			new Eraser(),
			new SolidBrush(selectedCellType, m_options.brushShade),
			new RandomizedBrush(selectedCellType),
			new TexturedBrush(selectedCellType, BitmapFileResource(m_options.brushTexturePath))
		};
	}

    void showImGuiWindow() {
        // Initialize window state and set its size
        bool isWindowOpen = true;
        bool c_initedSize = false;
        ImGui::Begin("GUI", &isWindowOpen, (c_initedSize) ? ImGuiWindowFlags_NoResize : 0);
        ImGui::SetWindowSize({ 0 , 0 });
        c_initedSize = true;

        // Disable camera movement when window is hovered
        m_cameraMovementDisabled = ImGui::IsWindowHovered();

        ImGui::SeparatorText("Simulation Speed:");
        showSimulationSpeedImGuiWidget();

        // Enable or disable chunk borders
        showChunkBorderEnableImGuiWidget();

        // Edit brush
        ImGui::SeparatorText("Brush:");
        m_brushSelectedIndex = Brush::selectBrushWithWidget(m_brushes, m_options, m_brushSelectedIndex);
        m_brush = m_brushes.at(m_brushSelectedIndex);

        // Display task times
        ImGui::SeparatorText("Performance:");
        PERF_MONITOR.showImGuiWidget();

        // Display mouse position
        ImGui::Separator();
        showMousePositionImGuiWidget();

        ImGui::End();
    }

	void moveCamera(vec2 mouseWorldPos, vec2 normalizedMousePos) {
        // Begin dragging if left mouse button is clicked
        static vec2 c_dragStart;
        static bool dragDisabled = false;
        if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
            dragDisabled = m_cameraMovementDisabled;
            c_dragStart = mouseWorldPos;
        }

        // Move camera while dragging
        if (IO::IsButtonDown(SDL_BUTTON_LEFT) && !dragDisabled) {
            m_options.camera.position = m_options.camera.position + (mouseWorldPos - c_dragStart);
        }

        // Handle zooming based on mouse wheel movement
        static Float zoom = 1.L;
        static Float zoomDP = 1.055;
        static Float zoomDN = 1.055;
        if (IO::GetMouseWheel() > 0)
            zoom = zoomDP;
        else if (IO::GetMouseWheel() < 0)
            zoom = 1.L / zoomDN;

        // Update properties considering zoom level and smooth zoom decrease
        m_options.camera.zoom *= zoom;

        auto& camPos = m_options.camera.position;
        if (abs(zoom - 1.L) > 0.0001f)
            camPos = camPos -
            ((normalizedMousePos) / m_options.camera.zoom - (normalizedMousePos) / (m_options.camera.zoom * zoom))
            ;

        zoom = 1.L + (zoom - 1.L) * 0.75L;
	}

    void showSimulationSpeedImGuiWidget() {
        ImGui::Text("Enabled: "); ImGui::SameLine(); ImGui::Checkbox("##simulationEnableCheckBox", &m_options.simulationEnabled);
        ImGui::Text("Min Update Delay: "); ImGui::SameLine(); ImGui::DragFloat("##minUpdateDelayDrag", &m_options.updateWaitTimeMs, 1, 0, 100);
        ImGui::Text("Min Render Delay: "); ImGui::SameLine(); ImGui::DragFloat("##minRenderDelayDrag", &m_options.renderWaitTimeMs, 1, 0, 100);
    }

    void showChunkBorderEnableImGuiWidget() {
        ImGui::Text("Chunk borders:"); ImGui::SameLine();
        ImGui::Checkbox("##showChunkBorders", &m_options.showChunkBorders);
    }

    void showMousePositionImGuiWidget() {
        vec2 mouseWorldPos = m_options.camera.screenToWorld(m_options.windowWidth, m_options.windowHeight, IO::GetMousePos());
        
        std::stringstream mousePosSStream;
        mousePosSStream << "Mouse position: {" << mouseWorldPos.x << "," << mouseWorldPos.y << "}";
        ImGui::Text(mousePosSStream.str().c_str());
    }

};
